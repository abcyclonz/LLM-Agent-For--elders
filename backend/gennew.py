def generate_response_node(state: AgentState) -> dict:
    """
    Generates a response using the most robust prompting patterns, including
    a check to prevent repeating alerts. This is the final, consolidated version.
    """
    global main_llm, AGENT_NAME
    print("--- Generate Response Node ---")
    user_input = state["user_input"]

    # --- START: No changes here, just getting data from state ---
    tool_result = state.get("tool_result")
    retrieved_context_str = state.get("retrieved_context")
    health_alerts = state.get("health_alerts")

    user_persona_data = state.get("user_persona", {})
    user_name = user_persona_data.get("name", "the user")
    # --- END: No changes here ---

    prompt_parts = []
    
    # --- START: NEW LOGIC BLOCK to determine if a health alert is new and needs to be forced ---
    # This replaces the simple `elif health_alerts:` logic.
    should_force_health_alert = False
    if health_alerts:
        # We check if the primary (most important) alert has already been mentioned.
        if not _has_alert_been_mentioned(health_alerts[0], state["messages"]):
            should_force_health_alert = True
    # --- END: NEW LOGIC BLOCK ---

    # --- START: RESTRUCTURED 'if/elif/else' block for clarity and correctness ---
    # This new structure correctly prioritizes the different modes of operation.
    
    if should_force_health_alert:
        # --- PRIORITY 1: Forced Health Alert Mode ---
        # This mode hijacks the conversation to deliver a critical, unmentioned alert.
        print("  >> Entering FORCED Health Alert Mode.")
        alerts_str = "\n- ".join(health_alerts)
        
        # We use the "Forced Context Injection" pattern by creating a fake HumanMessage.
        # This is the most reliable way to direct the LLM.
        forced_instruction = (
            f"(SYSTEM NOTE: You have detected new, critical health alerts. Your only goal for this turn is to address them. "
            f"Start your response by gently informing {user_name} about the following, then ask if they are okay. "
            f"Alerts to address: {alerts_str})"
        )
        
        system_prompt = (
            f"You are the '{AGENT_NAME}', a kind, patient, and empathetic AI companion. "
            "Your highest priority is the user's safety and well-being."
        )

        prompt_parts = [
            SystemMessage(content=system_prompt),
            *state["messages"], # Include all real past messages
            HumanMessage(content=forced_instruction) # Inject our command as the last message
        ]

    elif retrieved_context_str:
        # --- PRIORITY 2: Question Answering Mode ---
        # This runs if there's no new alert to force, but the router found memories to use.
        print("  >> Entering Persona-Aware Question-Answering Mode.")
        
        qa_prompt = (
            f"You are the '{AGENT_NAME}', a helpful and kind AI companion. You are speaking directly to your friend, {user_name}.\n\n"
            "**Your Task:**\n"
            f"You need to answer {user_name}'s question. The 'Context' below is **your own memory** - it contains facts you have learned about them in the past. Read their question, find the answer in your memory, and respond to them naturally in the first person ('I').\n\n"
            "**Rules:**\n"
            "1.  **Speak as 'I'.** For example, if the context says 'User is a history teacher', and the user asks 'what was my job?', you should answer 'I remember you telling me you were a history teacher.'\n"
            "2.  **Address them as 'you'.**\n"
            "3.  **Do not say 'based on the context' or refer to 'the user' in your response.** Treat the context as your own knowledge.\n"
            "4.  If the answer isn't in your memory, say something natural like 'I don't seem to recall that, I'm sorry.'\n\n"
            "--- CONTEXT (YOUR MEMORY) ---\n"
            f"{retrieved_context_str}\n"
            "---------------------------\n\n"
            f"--- {user_name.upper()}'S QUESTION ---\n"
            f"{user_input}\n"
            "------------------------\n\n"
            f"**YOUR RESPONSE TO {user_name.upper()}:**"
        )
        prompt_parts = [
            SystemMessage(content=qa_prompt)
        ]

    else:
        # --- PRIORITY 3: Standard Conversational Flow (Tool or General Chat) ---
        # This runs if there are no new alerts and no memories to retrieve.
        print("  >> Entering Standard Conversational Mode.")
        
        # Start with the base persona and user info.
        system_prompt_content = (
            f"You are the '{AGENT_NAME}', a kind, patient, and empathetic AI companion. "
            "Your primary role is to be a supportive and engaging conversational partner."
        )
        formatted_user_persona = format_persona_for_prompt(user_persona_data)
        if formatted_user_persona:
            system_prompt_content += f"\n\n--- User Information ---\n{formatted_user_persona}"

        # Define the mission for this turn (tool result or general chat).
        turn_specific_task = ""
        if tool_result:
            if "error" in tool_result.lower() or "failed" in tool_result.lower():
                turn_specific_task = (
                    "Your mission is to apologize and explain that a technical problem occurred. "
                    "Tell the user that you failed to complete the action due to a technical error with one of your tools (like the calendar tool). "
                    "Do not show them the raw error message. Just say something went wrong and you can't do it right now."
                )
            else:
                turn_specific_task = (
                    "Your mission is to confirm to the user that you have completed their request. "
                    "Speak naturally, as if you did it yourself. Do NOT mention a 'tool'.\n"
                    "For example, instead of 'The tool succeeded', say 'Okay, I've scheduled that for you.'\n\n"
                    f"Information to convey: '{tool_result}'"
                )
        else:
            turn_specific_task = (
                "Your mission is to be a good listener and conversational partner. "
                "Respond directly to the user's last message in a natural, engaging way."
            )
        
        # Append the mission to the system prompt.
        system_prompt_content += f"\n\n--- YOUR MISSION FOR THIS TURN ---\n{turn_specific_task}"
        
        # Build the final prompt parts for the LLM.
        prompt_parts = [SystemMessage(content=system_prompt_content.strip())]
        prompt_parts.extend(state["messages"])
        prompt_parts.append(HumanMessage(content=user_input))
    
    # --- END: RESTRUCTURED 'if/elif/else' block ---

    # --- START: No changes here, this invocation logic is correct ---
    try:
        response = main_llm.invoke(prompt_parts)
        ai_response_content = response.content
    except Exception as e:
        ai_response_content = f"I'm sorry, I encountered an error: {e}"

    print(f"  AI Response: {ai_response_content}")
    # We still add the original user_input to the official history
    updated_messages = add_messages(state["messages"], [HumanMessage(content=user_input), AIMessage(content=ai_response_content)])
    return {"messages": updated_messages, "user_input": ""}