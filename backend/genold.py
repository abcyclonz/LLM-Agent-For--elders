def generate_response_node(state: AgentState) -> dict:
    global main_llm, AGENT_NAME
    print("--- Generate Response Node ---")
    user_input = state["user_input"]

    tool_result = state.get("tool_result")
    retrieved_context_str = state.get("retrieved_context")
    health_alerts = state.get("health_alerts")

    # --- FIX: Get user_name dynamically from the state ---
    user_persona_data = state.get("user_persona", {})
    user_name = user_persona_data.get("name", "the user")

    system_prompt_content = (
        f"You are the '{AGENT_NAME}', a kind, patient, and empathetic AI companion. "
        "Your primary role is to be a supportive and engaging conversational partner."
    )
    
    formatted_user_persona = format_persona_for_prompt(user_persona_data)
    if formatted_user_persona:
        system_prompt_content += f"\n\n--- User Information ---\n{formatted_user_persona}"

    turn_specific_task = ""
    if retrieved_context_str:
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
        # --- FIX: The prompt_parts for QA mode should only contain the system message ---
        prompt_parts = [
            SystemMessage(content=qa_prompt)
        ]
    elif health_alerts:
        # --- NEW: Forced Health Alert Mode ---
        # In this mode, we IGNORE the user's latest message and force the conversation.
        print("  >> Entering FORCED Health Alert Mode.")
        alerts_str = "\n- ".join(health_alerts)
        
        # We construct a fake "Human" message that contains our instructions.
        # This is the most powerful way to direct the LLM.
        forced_instruction = (
            f"(SYSTEM NOTE: You have detected critical health alerts. Your only goal for this turn is to address them. "
            f"Start your response by gently informing {user_name} about the following, then ask if they are okay. "
            f"Alerts to address: {alerts_str})"
        )
        
        system_prompt = (
            f"You are the '{AGENT_NAME}', a kind, patient, and empathetic AI companion. "
            "Your highest priority is the user's safety and well-being."
        )

        prompt_parts = [
            SystemMessage(content=system_prompt),
            # We add the user's real messages from history
            *state["messages"], 
            # We inject our fake instruction as if the user just said it
            HumanMessage(content=forced_instruction)
        ]
        
    elif tool_result:
        # --- Tool Result Mode ---
        print("  >> Entering Tool Result Mode.")
        # ... (The existing mission-based approach is fine for tool results)
        system_prompt_content = (f"You are the '{AGENT_NAME}'...")
        # ... build the turn_specific_task for the tool result
        turn_specific_task = ( ... )
        system_prompt_content += f"\n\n--- YOUR MISSION FOR THIS TURN ---\n{turn_specific_task}"
        prompt_parts = [SystemMessage(content=system_prompt_content.strip())]
        prompt_parts.extend(state["messages"])
        prompt_parts.append(HumanMessage(content=user_input))
    else:
        # This is for all other conversational modes.
        print("  >> Entering General Conversational Mode.")
        if health_alerts:
            alerts_str = "\n- ".join(health_alerts)
            turn_specific_task = (
                "!!! URGENT HEALTH ALERT !!!\n"
                "Your most important mission is to gently inform the user about the following health observations. "
                "You MUST begin your response by addressing these points. This is your highest priority.\n"
                "Health Observations:\n- "
                f"{alerts_str}"
            )
        elif tool_result:
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

        system_prompt_content += f"\n\n--- YOUR MISSION FOR THIS TURN ---\n{turn_specific_task}"
        
        prompt_parts = [SystemMessage(content=system_prompt_content.strip())]
        prompt_parts.extend(state["messages"])
        prompt_parts.append(HumanMessage(content=user_input))

    try:
        response = main_llm.invoke(prompt_parts)
        ai_response_content = response.content
    except Exception as e:
        ai_response_content = f"I'm sorry, I encountered an error: {e}"

    print(f"  AI Response: {ai_response_content}")
    updated_messages = add_messages(state["messages"], [HumanMessage(content=user_input), AIMessage(content=ai_response_content)])
    return {"messages": updated_messages, "user_input": ""}