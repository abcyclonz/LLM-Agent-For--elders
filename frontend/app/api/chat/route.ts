// api/chat/route.ts
import { NextResponse } from "next/server";

export const maxDuration = 30;

export async function POST(req: Request) {
  try {
    const { messages, agentState } = await req.json();
    console.log("API route received:", { messages, agentState });

    if (!messages || messages.length === 0) {
      return NextResponse.json({ error: "No messages provided" }, { status: 400 });
    }

    // Get user token from localStorage (this will be passed from the frontend)
    const userToken = agentState?.userToken || null;

    const payload = {
      session_id: agentState?.sessionId || "frontend-session",
      user_input: messages[messages.length - 1]?.content || "",
      user_token: userToken,
    };
    console.log("Sending to backend:", payload);

    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      console.error("Backend error:", response.status, response.statusText);
      const errorText = await response.text();
      console.error("Backend error details:", errorText);
      return NextResponse.json(
        { error: `Backend error: ${response.status} ${response.statusText}`, details: errorText },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log("Backend response:", data);

    if (!data.ai_response) {
      console.error("Backend response missing ai_response:", data);
      return NextResponse.json({ error: "Invalid backend response: missing ai_response" }, { status: 500 });
    }

    // Update agent state with backend response (you'll pass this back to the frontend)
    const updatedAgentState = {
      ...agentState,
      sessionId: data.session_id,
      turnCount: data.turn_count || 0,
      routerDecision: data.current_router_decision || "",
      retrievedContext: data.retrieved_context_for_turn || "",
      episodicMemoryLog: data.episodic_memory_log || [],
      longTermMemoryLog: data.long_term_memory_log || [],
      healthAlerts: data.health_alerts_for_turn || [],
    };

    // Return a JSON response with the AI's message and the updated agent state
    return NextResponse.json({
      message: {
        role: 'assistant',
        content: data.ai_response,
      },
      agentState: updatedAgentState,
    });

  } catch (error) {
    console.error("API route error:", error);
    return NextResponse.json(
      { error: `Internal Server Error: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    );
  }
}