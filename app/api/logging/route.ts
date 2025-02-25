import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const LOG_FILE_PATH = path.join(process.cwd(), 'app/api/logging/log.json');

interface LogEntry {
  userId: string;
  timestamp: string;
  question: string;
}

async function readLogFile(): Promise<LogEntry[]> {
  try {
    const data = await fs.readFile(LOG_FILE_PATH, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    // If file doesn't exist or is empty, return empty array
    return [];
  }
}

async function writeLogFile(logs: LogEntry[]) {
  await fs.writeFile(LOG_FILE_PATH, JSON.stringify(logs, null, 2));
}

export async function POST(req: Request) {
  try {
    const logData: LogEntry = await req.json();
    
    // Validate required fields
    if (!logData.userId || !logData.timestamp || !logData.question) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Read existing logs
    const existingLogs = await readLogFile();
    
    // Add new log entry
    existingLogs.push(logData);
    
    // Write updated logs back to file
    await writeLogFile(existingLogs);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error logging data:', error);
    return NextResponse.json(
      { error: 'Failed to log data' },
      { status: 500 }
    );
  }
}

// Optional: Add GET endpoint to retrieve logs if needed
export async function GET() {
  try {
    const logs = await readLogFile();
    return NextResponse.json(logs);
  } catch (error) {
    console.error('Error reading logs:', error);
    return NextResponse.json(
      { error: 'Failed to read logs' },
      { status: 500 }
    );
  }
} 