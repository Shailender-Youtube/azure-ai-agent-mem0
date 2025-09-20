"""
Test script to verify if Mem0 can work with Azure AI Foundry Agent Service
"""
import os
from dotenv import load_dotenv
from mem0 import Memory
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

# Load environment variables from .env file
load_dotenv()

AZURE_AI_FOUNDRY_ENDPOINT = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
AZURE_AI_FOUNDRY_API_KEY = os.getenv("AZURE_AI_FOUNDRY_API_KEY")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME")
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
AGENT_ID = os.getenv("AGENT_ID")

# Azure AI Search config
AZURE_AI_SEARCH_SERVICE_NAME = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_AI_SEARCH_COLLECTION_NAME = os.getenv("AZURE_AI_SEARCH_COLLECTION_NAME")
EMBEDDING_MODEL_DIMS = int(os.getenv("EMBEDDING_MODEL_DIMS", "1536"))

# (Removed sample secrets/endpoints)

# Initialize Azure AI Project Client
project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint=PROJECT_ENDPOINT
)

# Mem0 configuration for Azure AI Foundry with Agent Service
memory_config = {
    "vector_store": {
        "provider": "azure_ai_search",
        "config": {
            "service_name": AZURE_AI_SEARCH_SERVICE_NAME,
            "api_key": AZURE_AI_SEARCH_API_KEY,
            "collection_name": AZURE_AI_SEARCH_COLLECTION_NAME,
            "embedding_model_dims": EMBEDDING_MODEL_DIMS,  # text-embedding-3-small dimension
        },
    },
    "embedder": {
        "provider": "azure_openai",
        "config": {
            "model": EMBEDDING_DEPLOYMENT_NAME,
            "embedding_dims": EMBEDDING_MODEL_DIMS,
            "azure_kwargs": {
                "api_version": "2023-05-15",
                "azure_deployment": EMBEDDING_DEPLOYMENT_NAME,
                "azure_endpoint": AZURE_AI_FOUNDRY_ENDPOINT,
                "api_key": AZURE_AI_FOUNDRY_API_KEY,
            },
        },
    },
    # For LLM, we'll still use Azure OpenAI directly since Mem0 doesn't natively support Azure AI Agents
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": CHAT_DEPLOYMENT_NAME,
            "temperature": 0.1,
            "max_tokens": 2000,
            "azure_kwargs": {
                "azure_deployment": CHAT_DEPLOYMENT_NAME,
                "api_version": "2025-01-01-preview",
                "azure_endpoint": AZURE_AI_FOUNDRY_ENDPOINT,
                "api_key": AZURE_AI_FOUNDRY_API_KEY,
            },
        },
    },
    "version": "v1.1",
}

class CookingAssistantWithMemory:
    """Recipe & Cooking Assistant with persistent memory using Mem0 and Azure AI Agent"""
    
    def __init__(self):
        # Initialize Mem0
        self.memory = Memory.from_config(memory_config)
        
        # Get the Azure AI Agent
        self.agent = project.agents.get_agent(AGENT_ID)
        print(f"🍳 Connected to your Personal Cooking Assistant: {self.agent.id}")
        print("I'll remember your cooking preferences, dietary needs, and recipe history!")
        
        # Define required user profile fields for onboarding completeness
        self.required_profile_fields = [
            "skill_level",
            "dietary_preferences",
            "allergies",
            "dislikes",
            "kitchen_equipment",
            "favorite_cuisines",
            "preferred_meal_types",
            "time_constraints"
        ]
        # Minimal fields to be "recipe-ready" without further onboarding
        self.min_profile_fields = [
            "skill_level",
            "dietary_preferences",
            "allergies",
        ]

    def _get_profile_status(self, user_id="default_user"):
        """Return a dict of {field: value} for PROFILE.* memories from the user's full memory list."""
        profile = {}
        # Read all and filter to avoid semantic search misses/latency
        all_items = self.memory.get_all(user_id=user_id).get('results', [])
        for item in all_items:
            text = item.get('memory', '')
            if isinstance(text, str) and text.startswith("PROFILE.") and ":" in text:
                try:
                    key_part, value_part = text.split(":", 1)
                    field = key_part.replace("PROFILE.", "").strip()
                    value = value_part.strip()
                    if field and value:
                        profile[field] = value
                except Exception:
                    continue
        return profile

    def _infer_profile_from_plain_text(self, user_id="default_user"):
        """Infer profile fields from free-text memories using simple heuristics. Returns (profile, confidence)."""
        inferred = {}
        confidence = {}
        all_items = self.memory.get_all(user_id=user_id).get('results', [])
        texts = [m.get('memory', '') for m in all_items if isinstance(m.get('memory', ''), str)]
        if not texts:
            return inferred, confidence
        joined = " \n ".join(texts).lower()
        # Skill level heuristics including common misspelling
        if any(k in joined for k in ["intermediate", "intermediatry", "intermediatery"]):
            inferred["skill_level"] = "intermediate"; confidence["skill_level"] = 0.8
        elif "beginner" in joined:
            inferred["skill_level"] = "beginner"; confidence["skill_level"] = 0.8
        elif "advanced" in joined:
            inferred["skill_level"] = "advanced"; confidence["skill_level"] = 0.8
        # Dietary
        if "vegetarian" in joined:
            inferred["dietary_preferences"] = "vegetarian"; confidence["dietary_preferences"] = 0.7
        if "vegan" in joined:
            inferred["dietary_preferences"] = "vegan"; confidence["dietary_preferences"] = 0.7
        # Allergies (examples)
        if "lactose" in joined:
            inferred["allergies"] = "lactose intolerance"; confidence["allergies"] = max(confidence.get("allergies",0.0), 0.7)
        if "gluten" in joined or "celiac" in joined:
            inferred["allergies"] = "gluten"; confidence["allergies"] = max(confidence.get("allergies",0.0), 0.7)
        for nut in ["peanut","peanuts","tree nut","almond","cashew","walnut"]:
            if nut in joined:
                inferred["allergies"] = "nuts"; confidence["allergies"] = max(confidence.get("allergies",0.0), 0.7)
        for allergen in ["shellfish","shrimp","prawn","crab","egg allergy","soy allergy","soy intolerant"]:
            if allergen in joined:
                inferred["allergies"] = allergen.split(" ")[0]; confidence["allergies"] = max(confidence.get("allergies",0.0), 0.7)
        # Dislikes
        for item in ["mushroom", "coriander", "olive", "capsicum"]:
            if f"dislike {item}" in joined or f"do not like {item}" in joined or f"don't like {item}" in joined:
                inferred.setdefault("dislikes", []).append(item)
                confidence["dislikes"] = max(confidence.get("dislikes", 0.0), 0.6)
        # Equipment
        for tool in ["air fryer", "slow cooker", "pressure cooker", "blender", "grill"]:
            if tool in joined:
                inferred.setdefault("kitchen_equipment", []).append(tool)
                confidence["kitchen_equipment"] = max(confidence.get("kitchen_equipment", 0.0), 0.6)
        return inferred, confidence

    def _merge_structured_and_inferred(self, structured, inferred, confidence, min_confidence=0.7):
        """Merge structured PROFILE.* values with inferred values using a confidence threshold."""
        merged = {}
        # Start with inferred above threshold
        for key, val in inferred.items():
            if confidence.get(key, 0.0) >= min_confidence:
                merged[key] = val
        # Override with structured where present
        for key, val in structured.items():
            merged[key] = val
        return merged

    def _first_missing_field_from(self, profile_dict):
        for field in self.required_profile_fields:
            if field not in profile_dict or not profile_dict[field]:
                return field
        return None

    def _first_missing_with_priority(self, profile_dict):
        # Prioritize minimal fields first
        for field in self.min_profile_fields:
            if field not in profile_dict or not profile_dict[field]:
                return field
        # Then any remaining optional fields
        for field in self.required_profile_fields:
            if field not in profile_dict or not profile_dict[field]:
                return field
        return None

    def _has_complete_profile(self, user_id="default_user"):
        profile = self._get_profile_status(user_id=user_id)
        return all(field in profile and profile[field] for field in self.required_profile_fields)

    def _parse_and_store_profile_tags(self, agent_response, user_id="default_user"):
        """Extract lines like 'PROFILE.field: value' from the agent response and store them as separate memories."""
        if not agent_response:
            return
        lines = agent_response.splitlines()
        for line in lines:
            if line.strip().startswith("PROFILE.") and ":" in line:
                self.memory.add(line.strip(), user_id=user_id)

    def _first_missing_field(self, user_id="default_user"):
        profile = self._get_profile_status(user_id=user_id)
        for field in self.required_profile_fields:
            if field not in profile or not profile[field]:
                return field
        return None

    def _maybe_capture_from_user_input(self, user_message, user_id="default_user"):
        """If onboarding is in progress and user_message likely answers the next question, store PROFILE.<field>.
        Returns (captured: bool, field: str|None, value: str|None)."""
        next_field = self._first_missing_field(user_id=user_id)
        if not next_field:
            return (False, None, None)
        text = (user_message or "").strip()
        if not text:
            return (False, None, None)
        # Minimal heuristic: accept user's reply as value for the expected field
        value = text
        # Light normalization for skill level
        if next_field == "skill_level":
            lowered = text.lower()
            # Only capture if the user explicitly provided one of the allowed levels.
            if "beginner" in lowered:
                value = "beginner"
            elif "intermediate" in lowered:
                value = "intermediate"
            elif "advanced" in lowered:
                value = "advanced"
            else:
                # Ignore unrelated replies (e.g., "medium spicy") to avoid corrupting skill_level
                return (False, None, None)
        profile_line = f"PROFILE.{next_field}: {value}"
        self.memory.add(profile_line, user_id=user_id)
        return (True, next_field, value)
    
    def chat_with_memory(self, user_message, user_id="default_user", thread=None):
        """Chat with the agent while using Mem0 for persistent memory. Requires a thread object."""
        summary_triggers = [
            "what do you know about me",
            "print all my memories",
            "summarize my memories",
            "list my memories",
            "what have I told you",
            "what do you remember about me"
        ]
        is_summary = any(trigger in user_message.lower() for trigger in summary_triggers)
        memory_context = ""
        use_memories = False
        if is_summary:
            print("\n🧠 Fetching ALL memories for summary...")
            all_memories = self.memory.get_all(user_id=user_id).get('results', [])
            if all_memories:
                memory_context = "\n\nHere is everything I remember about you from our past conversations:\n"
                for i, mem in enumerate(all_memories, 1):
                    memory_context += f"{i}. {mem['memory']}\n"
                print(f"   Found {len(all_memories)} total memories")
            else:
                memory_context = "\n\nI don't have any memories stored for you yet.\n"
            # Return memory listing directly without calling the agent
            return memory_context.strip()
        else:
            # Use semantic search for relevant memories, with a high limit
            print(f"\n🧠 Semantic search for relevant user memories...")
            relevant_memories = self.memory.search(user_message, user_id=user_id, limit=100)
            if relevant_memories.get('results'):
                memory_context = "\n\nRelevant facts about you from previous conversations:\n"
                for i, mem in enumerate(relevant_memories['results'], 1):
                    memory_context += f"{i}. {mem['memory']}\n"
                print(f"   Found {len(relevant_memories['results'])} relevant user memories")
                # Always include the latest user message as a fact
                memory_context += f"\n\nLatest user message: {user_message}\n"
                use_memories = True
            else:
                print("   No relevant user memories found")
        # Add cooking-specific instruction for the agent
        cooking_instruction = """
        COOKING ASSISTANT PERSONA: You are a friendly, enthusiastic cooking assistant who loves to chat! 
        
        PERSONALITY:
        - Be warm, encouraging, and conversational
        - Ask ONE clear question at a time to keep it interactive
        - Show excitement about cooking and food
        - Use cooking emojis and friendly language
        - Be curious about their cooking experiences
        - Offer helpful tips and encouragement
        
        REMEMBER & TRACK:
        - User's cooking skill level (beginner, intermediate, advanced)
        - Dietary restrictions/preferences (vegetarian, vegan, allergies, dislikes)
        - Kitchen equipment available
        - Favorite cuisines and flavors
        - Successful recipes they've tried
        - Cooking challenges they face
        - Preferred meal types (quick, healthy, comfort food, etc.)
        - RECIPES ALREADY SUGGESTED (don't repeat unless user specifically asks for same recipe)
        
        ALWAYS:
        - Suggest NEW recipes they haven't tried yet
        - Respect their dietary needs and preferences
        - Build on their previous successes with NEW variations
        - Offer fresh ideas based on what they have/like
        - Encourage and provide helpful cooking tips
        - Remember what worked or didn't work for them
        - AVOID repeating recipes you've already suggested unless they ask for the same one again
        - Ask engaging questions to learn more about their preferences
        - Be enthusiastic and supportive of their cooking journey
        
        INTERACTIVE STYLE:
        - Ask ONE SIMPLE question at a time - don't overwhelm with multiple questions
        - Wait for their answer before asking the next question
        - Show interest in their cooking results
        - Celebrate their successes
        - Offer encouragement for challenges
        - Suggest next steps or related recipes
        
        CRITICAL: Ask only ONE question per response to keep the conversation natural and interactive!
        Be encouraging, practical, personalized, and keep the conversation flowing one question at a time!
        """
        
        if use_memories:
            memory_context += cooking_instruction
        else:
            memory_context = cooking_instruction
        # Build merged profile view (structured + inferred) before adding onboarding guidance
        structured = self._get_profile_status(user_id=user_id)
        inferred, inferred_conf = self._infer_profile_from_plain_text(user_id=user_id)
        merged_profile = self._merge_structured_and_inferred(structured, inferred, inferred_conf)
        merged_complete = all(f in merged_profile and merged_profile[f] for f in self.required_profile_fields)

        # 2. Use the provided thread for this session
        if thread is None:
            raise ValueError("Thread must be provided for session chat.")
        # 3. Before sending to the agent, if onboarding is incomplete (merged view) build a targeted next question
        if not merged_complete:
            profile_now = merged_profile
            next_field = self._first_missing_field_from(merged_profile)
            if next_field:
                # If user asked general things, steer to next field directly
                targeted_prompt = (
                    f"Let's continue your profile. So far I have: "
                    f"{'; '.join([f'{k.replace('_',' ')}: {v}' for k,v in profile_now.items()]) or 'none yet'}. "
                    f"Please share your {next_field.replace('_',' ')}. "
                    f"Respond naturally; I'll record it as PROFILE.{next_field}: <value>."
                )
                user_message = targeted_prompt if len(user_message.strip()) == 0 or user_message.strip().endswith("?") else user_message

        enhanced_message = f"{user_message}{memory_context}"
        # If user likely answered the next onboarding question, capture immediately and reflect it in the message
        captured, captured_field, captured_value = (False, None, None)
        if not merged_complete:
            captured, captured_field, captured_value = self._maybe_capture_from_user_input(user_message, user_id=user_id)
            if captured and captured_field and captured_value:
                enhanced_message += f"\n\nPROFILE.{captured_field}: {captured_value}"
        # Add onboarding guidance if profile is incomplete
        current_profile = self._get_profile_status(user_id=user_id)
        if not merged_complete and not captured:
            next_field = self._first_missing_field_from(merged_profile)
            profile_bits = []
            for key in self.required_profile_fields:
                if key in merged_profile:
                    profile_bits.append(f"{key.replace('_', ' ')}: {merged_profile[key]}")
            profile_summary = "; ".join(profile_bits) if profile_bits else "none yet"
            onboarding_guidance = (
                f"\n\nONBOARDING GUIDANCE (do not show to user): "
                f"Collected profile so far -> {profile_summary}. "
                f"Ask ONLY about '{next_field.replace('_', ' ')}' next. "
                f"When the user answers, include a new line 'PROFILE.{next_field}: <value>' in your reply. "
                f"Do not repeat questions that have already been answered."
            )
            enhanced_message += onboarding_guidance
        message = project.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=enhanced_message
        )
        # 4. Run the agent
        print(f"🤖 Running Azure AI Agent...")
        run = project.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=self.agent.id
        )
        if run.status == "failed":
            print(f"❌ Agent run failed: {run.last_error}")
            return None
        # 5. Get the LATEST assistant response (avoid stale replies)
        messages = project.agents.messages.list(
            thread_id=thread.id,
            order=ListSortOrder.DESCENDING
        )
        agent_response = None
        for message in messages:
            if message.role == "assistant" and message.text_messages:
                # In DESC order, the first text message is the most recent chunk
                agent_response = message.text_messages[0].text.value
                break
        if agent_response:
            print(f"✅ Agent response received")
            # 6. Store the conversation in Mem0 for future reference
            conversation = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": agent_response}
            ]
            self.memory.add(conversation, user_id=user_id)
            print(f"💾 Conversation stored in memory")
            
            # 6b. Extract PROFILE.* tags (if any) and store them individually for structured recall
            self._parse_and_store_profile_tags(agent_response, user_id=user_id)
            # Self-heal: if assistant explicitly states skill level in natural language, store it
            try:
                lower = agent_response.lower()
                if "skill level" in lower:
                    if "beginner" in lower:
                        self.memory.add("PROFILE.skill_level: beginner", user_id=user_id)
                    elif "intermediate" in lower:
                        self.memory.add("PROFILE.skill_level: intermediate", user_id=user_id)
                    elif "advanced" in lower:
                        self.memory.add("PROFILE.skill_level: advanced", user_id=user_id)
            except Exception:
                pass
            # 6c. If the agent forgot to emit PROFILE.* but the user likely provided an answer, capture it
            if not self._has_complete_profile(user_id=user_id) and not captured:
                self._maybe_capture_from_user_input(user_message, user_id=user_id)
            
            # 7. Extract and store any recipe suggestions separately for tracking
            # Only track recipe suggestions after onboarding is complete
            if self._has_complete_profile(user_id=user_id) and any(keyword in agent_response.lower() for keyword in ['recipe', 'try ', 'make ', 'cook ', 'dish', 'meal']):
                # Store that this recipe was suggested to avoid repetition
                recipe_memory = f"Recipe suggested: {agent_response[:150]}..."
                self.memory.add(recipe_memory, user_id=user_id)
                print(f"🍳 Recipe suggestion tracked in memory")
            
            return agent_response
        else:
            print("❌ No response from agent")
            return None
    
    def get_all_memories(self, user_id="default_user"):
        """Get all stored memories for a user"""
        all_memories = self.memory.get_all(user_id=user_id)
        return all_memories.get('results', [])

def test_cooking_assistant():
    """Test the Cooking Assistant with Mem0 memory integration"""
    
    try:
        print("=" * 60)
        print("🍳 Personal Cooking Assistant with Memory")
        print("=" * 60)
        cooking_assistant = CookingAssistantWithMemory()
        print("✅ Your cooking assistant is ready!")
        
        print("🍳 Welcome to your Personal Cooking Assistant!")
        print("I'm here to help you on your culinary journey!")
        print("Type 'exit' to end our cooking session.")
        
        user_id = input("👋 What's your name? I'd love to get to know you: ").strip()
        if user_id.lower() == "exit":
            print("Happy cooking!")
            return True
        
        # Create a single thread for this cooking session
        thread = project.agents.threads.create()
        print(f"🧑‍🍳 Started your personal cooking session: {thread.id}")
        
        # Decide onboarding vs. direct cooking using merged (structured + inferred) profile
        structured = cooking_assistant._get_profile_status(user_id=user_id)
        inferred, inferred_conf = cooking_assistant._infer_profile_from_plain_text(user_id=user_id)
        merged_profile = cooking_assistant._merge_structured_and_inferred(structured, inferred, inferred_conf)
        merged_complete = all(f in merged_profile and merged_profile[f] for f in cooking_assistant.required_profile_fields)

        if merged_complete:
            profile_bits = []
            for key in cooking_assistant.required_profile_fields:
                if key in merged_profile:
                    profile_bits.append(f"{key.replace('_', ' ')}: {merged_profile[key]}")
            profile_summary = "; ".join(profile_bits)
            welcome_msg = (
                f"Welcome back, {user_id}! I have your cooking profile on file ({profile_summary}). "
                f"Are you looking for a recipe now? If you have specific ingredients, tell me; "
                f"otherwise I can suggest something great for you."
            )
            initial_response = cooking_assistant.chat_with_memory(welcome_msg, user_id=user_id, thread=thread)
            print(f"🍳 Chef Assistant: {initial_response}\n")
        else:
            # Ask only for the next missing field using merged view
            next_field = cooking_assistant._first_missing_field_from(merged_profile)
            have_bits = []
            for key in cooking_assistant.required_profile_fields:
                if key in merged_profile:
                    have_bits.append(f"{key.replace('_', ' ')}: {merged_profile[key]}")
            have_summary = "; ".join(have_bits) if have_bits else "none yet"
            targeted_prompt = (
                f"Welcome back, {user_id}! We'll quickly complete your profile so I can tailor recipes. "
                f"So far I have -> {have_summary}. "
                f"Please share your {next_field.replace('_',' ')}. "
                f"Answer naturally; I'll remember it. Once done, I can suggest recipes or use your ingredients."
            )
            initial_response = cooking_assistant.chat_with_memory(targeted_prompt, user_id=user_id, thread=thread)
            print(f"🍳 Chef Assistant: {initial_response}\n")
        
        while True:
            user_message = input(f"\n{user_id}: ").strip()
            if user_message.lower() == "exit":
                break
            
            response = cooking_assistant.chat_with_memory(user_message, user_id=user_id, thread=thread)
            print(f"🍳 Chef Assistant: {response}")
            
        print(f"\nHappy cooking, {user_id}! I'll remember everything for next time! 👨‍🍳")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_cooking_assistant()
    
    if success:
        print("\n✅ COOKING SESSION COMPLETE!")
        print("   🍳 Your cooking assistant remembered everything from this session")
        print("   📝 All your preferences, skills, and recipe history are saved")
        print("   🚀 Next time you chat, I'll know exactly where we left off!")
    else:
        print("\n❌ COOKING SESSION FAILED: Check your credentials and configuration.")