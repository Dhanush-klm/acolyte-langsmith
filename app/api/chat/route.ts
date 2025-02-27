import { createMem0, addMemories } from '@mem0/vercel-ai-provider';
import { streamText } from 'ai';
import { Pool } from 'pg';
import { OpenAI } from 'openai';
import { wrapOpenAI } from 'langsmith/wrappers';
import { traceable } from 'langsmith/traceable';
import { AISDKExporter } from 'langsmith/vercel';

// Initialize OpenAI client with LangSmith wrapper
const openai = wrapOpenAI(new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
}));

// Initialize PostgreSQL connection pool
const pool = new Pool({
  connectionString: "postgresql://neondb_owner:2TYvAzNlt0Oy@ep-noisy-shape-a5hfgfjr.us-east-2.aws.neon.tech/documents?sslmode=require",
});

const mem0 = createMem0({
  provider: 'openai',
  mem0ApiKey: process.env.MEM0_API_KEY,
  apiKey: process.env.OPENAI_API_KEY,
  config: {
    compatibility: 'strict',
  },
});

interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

type Persona = 'general' | 'roleplay';

// System prompts for different personas
const SYSTEM_PROMPTS: Record<Persona, string> = {
  general: `You are a specialized assistant with the following guidelines:

1. Conversational Approach:
   - Maintain a friendly and natural dialog flow
   - Use a warm, approachable tone
   - Show genuine interest in user questions
   - Engage in a way that encourages continued conversation

2. Content Restrictions:
   - Base all responses strictly on the provided context and conversation history
   - Do not use any external knowledge
   - Avoid making assumptions beyond what is explicitly stated
   - Format numerical data and statistics exactly as they appear in the context

3. Response Guidelines:
   - When information is available: Provide accurate answers while maintaining a conversational tone
   - When information is missing: Say "I wish I could help with that, but I don't have enough information in the provided documentation to answer your question. Is there something else you'd like to know about?"
   - For follow-up questions: Verify that previous responses were based on documented content

4. Quality Standards:
   - Ensure accuracy while remaining approachable
   - Balance professionalism with conversational friendliness
   - Maintain consistency in information provided
   - Keep responses clear and engaging`,

  roleplay: `
This Teach-Back is an activity where the user practices a skill they just learned in an online course. Refer to the course storyboard as well as the course assessment to provide you with context. This activity will be scored and should reference only the material in the uploaded documents. You may reference other material in your feedback, but the scoring should be based solely on the course content. This activity is in section 2.2 of course 103. I have outlined how the activity is structured below. 
When the user clicks "begin," briefly describe the activity as a teach-back in which they'll receive personalized feedback based on their answer. Also, state the two rubric areas (Comprehensiveness and Clarity & Structure, each accounting for 4 points) and what a passing score is. Then, show the question: "Explain how drug pricing benchmarks impact pharmacy costs and reimbursement, and why this matters in pharmacy benefits consulting." 
After they submit their answer, grade them based on the rubric below and show them their score in each rubric area, along with what could be improved. Continue providing guidance to refine their answer until they achieve a score of 8/8, then summarize their response into a final statement and congratulate them. Instruct them to proceed in the course. 
When a user clicks "instructions," explain in detail how the activity works and highlight that they are aiming for mastery, and you will support them in achieving it. Show the full rubric and what their response should include (the 3 bullets below). 
The user's response should include: 
✔ A clear explanation of key drug pricing benchmarks (AWP, WAC, MAC, NADAC) and how they function. 
✔ An analysis of how these benchmarks influence pharmacy costs and reimbursement structures. 
✔ A connection to pharmacy benefits consulting, including how understanding benchmarks supports cost management and plan design. 
Evaluation Criteria: The user's response will be scored based on the rubric below, with a total of 8 possible points. To pass, they need at least 6 points. 
Scoring Rubric (8 Points Total) 
Comprehensiveness 
•	4: Clearly defines key drug pricing benchmarks, explains their role in pharmacy costs and reimbursement, and connects them to pharmacy benefits consulting. 
•	3: Mentions key drug pricing benchmarks and cost impact but lacks a full explanation or consulting connection. 
•	2: Provides a vague or incomplete definition of drug pricing benchmarks with little explanation of cost impact or relevance to consulting. 
•	1: Response is unclear, incorrect, or missing key details. 
Clarity & Structure 
•	4: Explanation is clear, well-organized, and easy to follow. 
•	3: Mostly clear but could be better structured or more concise. 
•	2: Somewhat unclear or disorganized. 
•	1: Hard to follow or confusing. 
✅ Passing Score: 6+ out of 8 
Example Response:  
Drug pricing benchmarks are essential in pharmacy benefits consulting as they determine how pharmacies are reimbursed and influence overall drug costs. AWP (Average Wholesale Price) is a benchmark for estimating drug prices, though often inflated. WAC (Wholesale Acquisition Cost) is the manufacturer's list price before rebates. MAC (Maximum Allowable Cost) limits reimbursement for generics, while NADAC (National Average Drug Acquisition Cost) reflects actual pharmacy costs. Consultants use these benchmarks to negotiate pricing, optimize formulary management, and control plan costs effectively.
`
};

// Wrap getQueryEmbedding with traceable
const getQueryEmbedding = traceable(async (query: string): Promise<number[]> => {
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: query,
  });
  return response.data[0].embedding;
}, {
  name: "Query Embedding Generation",
  metadata: {
    environment: process.env.VERCEL_ENV || 'development',
    model: "text-embedding-ada-002"
  }
});

// Wrap findSimilarContent with traceable
const findSimilarContent = traceable(async (embedding: number[]): Promise<string> => {
  // Format the embedding array as a PostgreSQL vector string
  const vectorString = `[${embedding.join(',')}]`;
  
  const query = `
    SELECT contents, 1 - (vector <=> $1::vector) as similarity
    FROM documents
    WHERE 1 - (vector <=> $1::vector) > 0.7
    ORDER BY similarity DESC
    LIMIT 5;
  `;
  
  const result = await pool.query(query, [vectorString]);
  return result.rows.map(row => row.contents).join('\n\n');
}, {
  name: "Similar Content Retrieval",
  metadata: {
    environment: process.env.VERCEL_ENV || 'development',
    dataSource: 'postgres',
    similarityThreshold: 0.7,
    maxResults: 5
  }
});

// Add tracing for message processing
const processMessages = traceable(async (
  messages: Message[], 
  userQuery: string, 
  persona: Persona, 
  embedding: number[],
  similarContent: string,
  userId: string
) => {
  const systemPrompt = `${SYSTEM_PROMPTS[persona]}

Documentation Context: ${similarContent}`;

  const updatedMessages = [
    { role: 'system' as const, content: systemPrompt },
    ...messages.slice(0, -1).map((msg: Message) => ({
      role: msg.role as 'system' | 'user' | 'assistant',
      content: msg.content
    })),
    { role: 'user' as const, content: userQuery }
  ];

  const result = await streamText({
    model: mem0('gpt-4o-mini', {
      user_id: userId,
    }),
    messages: updatedMessages,
    experimental_telemetry: AISDKExporter.getSettings({
      runName: 'chat-streaming',
      metadata: { 
        persona,
        messageCount: messages.length,
        userId
      }
    })
  });

  // Store the messages for memory
  await addMemories(updatedMessages as any, {
    user_id: userId,
    mem0ApiKey: process.env.MEM0_API_KEY,
  });

  return { result, updatedMessages };
}, {
  name: "Message Processing",
  metadata: {
    environment: process.env.VERCEL_ENV || 'development',
    runtime: 'edge'
  }
});

// Add greeting handler with tracing
const handleGreeting = traceable(async (
  userQuery: string,
  previousMessages: Message[],
  persona: Persona,
  userId: string
) => {
  const greetingResponse = getGreetingResponse();
  
  const result = await streamText({
    model: mem0('gpt-4o-mini', {
      user_id: userId,
    }),
    messages: [
      {
        role: 'system',
        content: SYSTEM_PROMPTS[persona],
      },
      ...previousMessages,
      {
        role: 'user',
        content: userQuery
      },
      {
        role: 'assistant',
        content: greetingResponse
      }
    ],
    experimental_telemetry: AISDKExporter.getSettings({
      runName: 'greeting-response',
      metadata: { 
        responseType: 'greeting',
        userId
      }
    })
  });

  // Add greeting to memories
  const greetingMessages = [
    { role: 'user', content: userQuery },
    { role: 'assistant', content: greetingResponse }
  ];
  await addMemories([...previousMessages, ...greetingMessages] as any, {
    user_id: userId,
    mem0ApiKey: process.env.MEM0_API_KEY,
  });

  return result;
}, {
  name: "Greeting Handler",
  metadata: {
    environment: process.env.VERCEL_ENV || 'development',
    responseType: 'greeting'
  }
});

// Function to check if message is a greeting
function isGreeting(query: string): boolean {
  const greetingPatterns = [
    /^(hi|hello|hey|good morning|good afternoon|good evening|greetings)(\s|$)/i,
    /^(how are you|what's up|wassup|sup)(\?|\s|$)/i,
    /^(hola|bonjour|hallo|ciao)(\s|$)/i
  ];
  
  return greetingPatterns.some(pattern => pattern.test(query.trim().toLowerCase()));
}

// Function to get greeting response
function getGreetingResponse(): string {
  const greetings = [
    "👋 Hello! How can I assist you today?",
    "Hi there! 😊 What can I help you with?",
    "👋 Hey! Ready to help you with any questions!",
    "Hello! 🌟 How may I be of assistance?",
    "Hi! 😃 Looking forward to helping you today!"
  ];
  
  return greetings[Math.floor(Math.random() * greetings.length)];
}

export const maxDuration = 30;

function isValidPersona(persona: any): persona is Persona {
  return ['general', 'roleplay'].includes(persona);
}

// Simple store for temporary data - just like in links API
let tempStore: {
  query?: string;
  similarContent?: string;
  userId?: string;
} = {};

// Traceable function to log the complete conversation
const traceCompleteConversation = traceable(async (query: string, context: string, answer: string) => {
  console.log('🔍 Tracing complete conversation:');
  console.log(`📝 Query: "${query.substring(0, 100)}..."`);
  console.log(`📚 Context length: ${context.length} characters`);
  console.log(`💬 Answer length: ${answer.length} characters`);
  
  return { 
    query, 
    contextLength: context.length, 
    answerLength: answer.length 
  };
}, {
  name: "Complete Conversation Trace",
  metadata: {
    environment: process.env.VERCEL_ENV || 'development'
  }
});

// Add this new traceable function to log all three components
const logFinalConversationState = traceable(async (query: string, context: string, answer: string) => {
  console.log('\n📊 FINAL CONVERSATION COMPONENTS:');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📝 USER QUERY:');
  console.log(query);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📚 CONTEXT:');
  console.log(context.length > 300 ? context.substring(0, 300) + '...' : context);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('💬 AI ANSWER:');
  console.log(answer);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
  
  return { 
    query, 
    contextLength: context.length, 
    answerLength: answer.length,
    answerSample: answer.substring(0, 100) + '...'
  };
}, {
  name: "Final Conversation State Log"
});

// Add a traceable function just for the answer
const traceAnswer = traceable(async (answer: string) => {
  console.log('💬 Tracing answer:');
  console.log(`- Length: ${answer.length} characters`);
  console.log(`- Preview: "${answer.substring(0, 100)}..."`);
  
  // You can do additional answer analysis here if needed
  const sentenceCount = answer.split(/[.!?]+/).length - 1;
  const wordCount = answer.split(/\s+/).length;
  
  return {
    answerLength: answer.length,
    sentenceCount,
    wordCount,
    previewText: answer.substring(0, 150)
  };
}, {
  name: "AI Answer Analysis",
  metadata: {
    component: 'answer-analysis',
    analysisType: 'text-metrics'
  }
});

export async function POST(req: Request) {
  try {
    const body = await req.json();
    
    // Case 1: Frontend sending back the complete answer
    if (body.completeAnswer) {
      console.log('\n=== Processing Complete Answer ===');
      console.log('💬 Answer received from frontend');
      
      // Check if we have stored query and context
      if (!tempStore.query || !tempStore.similarContent) {
        console.log('⚠️ No stored query or context found');
        return new Response(JSON.stringify({ 
          status: 'error', 
          message: 'No stored query found' 
        }));
      }
      
      console.log('✅ Found stored data:');
      console.log(`- Query: "${tempStore.query.substring(0, 50)}..."`);
      console.log(`- Context length: ${tempStore.similarContent.length} characters`);
      
      // Trace just the answer
      await traceAnswer(body.completeAnswer);
      
      // Trace the complete conversation (all three components)
      await traceCompleteConversation(
        tempStore.query, 
        tempStore.similarContent, 
        body.completeAnswer
      );
      
      // Add additional detailed logging of all three components
      await logFinalConversationState(
        tempStore.query,
        tempStore.similarContent,
        body.completeAnswer
      );
      
      console.log('✅ Successfully traced and logged conversation');
      
      // Clear the temporary store
      tempStore = {};
      console.log('🧹 Temporary store cleared');
      console.log('=== End Processing Complete Answer ===\n');
      
      return new Response(JSON.stringify({ status: 'traced' }));
    }
    
    // Case 2: Initial request - process and store
    const { messages, userId, persona: requestPersona } = body;
    const userQuery = messages[messages.length - 1].content;
    const persona = requestPersona || 'general';
    
    console.log('\n=== Processing Initial Request ===');
    console.log(`📝 New query: "${userQuery.substring(0, 100)}..."`);

    // Generate embeddings and get similar content
    const embedding = await getQueryEmbedding(userQuery);
    const similarContent = await findSimilarContent(embedding);
    console.log(`📚 Retrieved similar content: ${similarContent.length} characters`);
    
    // Store query and context for later tracing
    tempStore = {
      query: userQuery,
      similarContent: similarContent,
      userId: userId
    };
    console.log('💾 Stored query and context for later tracing');
    
    // Process messages and stream response
    const { result } = await processMessages(
      messages, 
      userQuery, 
      persona, 
      embedding, 
      similarContent,
      userId
    );
    
    console.log('✅ Request processed, streaming response');
    console.log('=== End Processing Initial Request ===\n');
    
    return new Response(result.toDataStreamResponse().body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      }
    });
  } catch (error) {
    console.error('❌ Error processing request:', error);
    // Clear tempStore in case of error
    tempStore = {};
    return new Response(JSON.stringify({ status: 'error', message: 'An error occurred' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}