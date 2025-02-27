import { NextResponse } from 'next/server';

// Return success response without actually logging anything
export async function POST(req: Request) {
  return NextResponse.json({ success: true });
}

// Optional: Leave a minimal GET endpoint that returns empty array
export async function GET() {
  return NextResponse.json([]);
} 