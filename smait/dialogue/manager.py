"""
SMAIT HRI System v2.0 - Dialogue Manager
Features:
- Sliding window conversation memory
- Multiple LLM backend support (OpenAI, Ollama, Anthropic)
- Noisy ASR handling in prompts
- Session-aware memory management
"""

import os
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Load .env file FIRST before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[ENV] Loaded .env file")
except ImportError:
    print("[ENV] python-dotenv not installed, using system environment variables")

from smait.core.config import get_config, LLMBackend
from smait.core.events import DialogueTurn, DialogueResponse


class ConversationMemory:
    """
    Sliding window conversation memory.
    Keeps the last N turns and optionally summarizes older context.
    """
    
    def __init__(
        self,
        system_prompt: str,
        max_turns: int = 10,
        enable_summarization: bool = False
    ):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.enable_summarization = enable_summarization
        
        self.turns: List[DialogueTurn] = []
        self.summary: Optional[str] = None
        self.session_start: float = time.time()
    
    def add_user_turn(self, text: str, confidence: float = 1.0):
        """Add user message to history"""
        turn = DialogueTurn(
            role="user",
            content=text,
            timestamp=time.time(),
            confidence=confidence
        )
        self.turns.append(turn)
        self._maybe_trim()
    
    def add_assistant_turn(self, text: str, latency_ms: float = 0.0):
        """Add assistant response to history"""
        turn = DialogueTurn(
            role="assistant",
            content=text,
            timestamp=time.time(),
            latency_ms=latency_ms
        )
        self.turns.append(turn)
        self._maybe_trim()
    
    def _maybe_trim(self):
        """Trim old turns if exceeding max"""
        # Count user+assistant pairs
        if len(self.turns) > self.max_turns * 2:
            # Remove oldest turn pair
            if self.enable_summarization:
                # TODO: Implement summarization of removed turns
                pass
            self.turns = self.turns[2:]  # Remove oldest pair
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in LLM format"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add summary if available
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        # Add turns
        for turn in self.turns:
            messages.append({
                "role": turn.role,
                "content": turn.content
            })
        
        return messages
    
    def reset(self):
        """Clear conversation history (new session)"""
        self.turns = []
        self.summary = None
        self.session_start = time.time()
    
    @property
    def turn_count(self) -> int:
        """Number of conversation turns (user messages)"""
        return sum(1 for t in self.turns if t.role == "user")


class LLMClient(ABC):
    """Abstract LLM client interface"""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> DialogueResponse:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    async def generate_async(self, messages: List[Dict[str, str]]) -> DialogueResponse:
        """Generate response asynchronously"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client"""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 150,
        temperature: float = 0.7
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            
            if api_key:
                self.client = OpenAI(api_key=api_key)
                print(f"[LLM] OpenAI client initialized (model={self.model})")
            else:
                print("[LLM] No OPENAI_API_KEY found!")
        except ImportError:
            print("[LLM] openai package not installed")
    
    def generate(self, messages: List[Dict[str, str]]) -> DialogueResponse:
        """Generate response synchronously"""
        if self.client is None:
            return DialogueResponse(
                text="[LLM not configured]",
                latency_ms=0,
                model=self.model
            )
        
        start = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            text = response.choices[0].message.content
            latency = (time.time() - start) * 1000
            
            return DialogueResponse(
                text=text,
                latency_ms=latency,
                token_count=response.usage.total_tokens if response.usage else 0,
                model=self.model
            )
            
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return DialogueResponse(
                text=f"[Error: {str(e)}]",
                latency_ms=(time.time() - start) * 1000,
                model=self.model
            )
    
    async def generate_async(self, messages: List[Dict[str, str]]) -> DialogueResponse:
        """Generate response asynchronously"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, messages)


class OllamaClient(LLMClient):
    """Ollama local LLM client"""
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 150,
        temperature: float = 0.7
    ):
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"[LLM] Ollama client initialized (model={self.model})")
    
    def generate(self, messages: List[Dict[str, str]]) -> DialogueResponse:
        """Generate response using Ollama API"""
        import requests
        
        start = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": self.max_tokens,
                        "temperature": self.temperature
                    }
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            text = data.get("message", {}).get("content", "")
            latency = (time.time() - start) * 1000
            
            return DialogueResponse(
                text=text,
                latency_ms=latency,
                model=self.model
            )
            
        except Exception as e:
            print(f"[LLM] Ollama error: {e}")
            return DialogueResponse(
                text=f"[Error: {str(e)}]",
                latency_ms=(time.time() - start) * 1000,
                model=self.model
            )
    
    async def generate_async(self, messages: List[Dict[str, str]]) -> DialogueResponse:
        """Generate response asynchronously"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, messages)


class DialogueManager:
    """
    High-level dialogue management.
    Handles conversation flow, memory, and LLM interaction.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Create LLM client
        if self.config.dialogue.llm_backend == LLMBackend.OPENAI:
            self.llm = OpenAIClient(
                model=self.config.dialogue.model_name,
                max_tokens=self.config.dialogue.max_tokens,
                temperature=self.config.dialogue.temperature
            )
        elif self.config.dialogue.llm_backend == LLMBackend.OLLAMA:
            self.llm = OllamaClient(
                model=self.config.dialogue.model_name,
                max_tokens=self.config.dialogue.max_tokens,
                temperature=self.config.dialogue.temperature
            )
        else:
            # Default to OpenAI
            self.llm = OpenAIClient()
        
        # Create conversation memory
        self.memory = ConversationMemory(
            system_prompt=self.config.dialogue.system_prompt,
            max_turns=self.config.dialogue.max_history_turns,
            enable_summarization=self.config.dialogue.enable_summarization
        )
        
        # Callbacks
        self._on_response: Optional[Callable[[DialogueResponse], None]] = None
    
    def set_response_callback(self, callback: Callable[[DialogueResponse], None]):
        """Set callback for responses"""
        self._on_response = callback
    
    def ask(self, text: str, confidence: float = 1.0) -> DialogueResponse:
        """
        Process user input and generate response.
        
        Args:
            text: User's message (from ASR)
            confidence: ASR confidence score
        
        Returns:
            DialogueResponse with assistant's reply
        """
        # Add user turn to memory
        self.memory.add_user_turn(text, confidence)
        
        # Get messages for LLM
        messages = self.memory.get_messages()
        
        # Generate response
        response = self.llm.generate(messages)
        
        # Add assistant turn to memory
        self.memory.add_assistant_turn(response.text, response.latency_ms)
        
        # Callback
        if self._on_response:
            self._on_response(response)
        
        return response
    
    async def ask_async(self, text: str, confidence: float = 1.0) -> DialogueResponse:
        """Async version of ask()"""
        # Add user turn to memory
        self.memory.add_user_turn(text, confidence)
        
        # Get messages for LLM
        messages = self.memory.get_messages()
        
        # Generate response
        response = await self.llm.generate_async(messages)
        
        # Add assistant turn to memory
        self.memory.add_assistant_turn(response.text, response.latency_ms)
        
        # Callback
        if self._on_response:
            self._on_response(response)
        
        return response
    
    def reset_session(self):
        """Reset conversation memory (new session)"""
        self.memory.reset()
        print("[DIALOGUE] Session reset")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.memory.get_messages()
    
    @property
    def turn_count(self) -> int:
        """Number of conversation turns"""
        return self.memory.turn_count


# Default system prompts for different robot personas
SYSTEM_PROMPTS = {
    "luma": (
        "You are 'Luma', a friendly human-like service robot assistant. "
        "Keep responses short (2-5 sentences) unless asked for detail. "
        "Speak naturally and warmly, never robotic or overly formal. "
        "Avoid phrases like 'As an AI model' or 'I don't have feelings.' "
        "\n\n"
        "NOISY ASR HANDLING: User messages may contain background voices, "
        "random extra words, or partial phrases from imperfect speech recognition. "
        "Focus on the main intent and ignore obviously unrelated fragments. "
        "If truly unintelligible, politely ask for clarification."
    ),
    
    "concierge": (
        "You are a professional hotel concierge robot. "
        "Provide helpful information about the hotel, local attractions, and services. "
        "Be polite, efficient, and informative. "
        "Keep responses concise unless detailed information is requested."
    ),
    
    "healthcare": (
        "You are a healthcare assistant robot in a hospital or clinic setting. "
        "Be empathetic, clear, and reassuring. "
        "Provide general information but always recommend consulting with medical professionals. "
        "Maintain patient privacy and confidentiality."
    ),
    
    "retail": (
        "You are a retail assistant robot in a store. "
        "Help customers find products, provide information, and assist with navigation. "
        "Be friendly and helpful, but not pushy. "
        "Acknowledge when you need to check with a human associate."
    )
}


def create_dialogue_manager(persona: str = "luma") -> DialogueManager:
    """Factory function to create dialogue manager with specific persona"""
    config = get_config()
    
    if persona in SYSTEM_PROMPTS:
        config.dialogue.system_prompt = SYSTEM_PROMPTS[persona]
    
    return DialogueManager()
