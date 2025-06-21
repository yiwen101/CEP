"""
Shared CEP Prompts for Context Elaboration
Contains prompts for different categories and domains
"""

from typing import Dict, List

class CEPPrompts:
    """Context Elaboration Prompts for different categories and domains"""
    
    def __init__(self):
        # General CEP prompts (from research proposal)
        self.general_ceps = {
            "understand": [
                "Paraphrase the provided information in your own words",
                "Summarize the given text in a clear and concise manner"
            ],
            "connect": [
                "What does this information remind you of? Briefly explain the connection.",
                "How does this information relate to other facts or concepts you know?"
            ],
            "query": [
                "What do you find to be the most surprising or interesting piece of information?",
                "Formulate two insightful questions that are raised by the text"
            ],
            "application": [
                "What can you deduce from the given information?",
                "Formulate two insightful questions that are answered by the information given"
            ],
            "comprehensive": [
                """Please elaborate on the provided context by:
                1. Paraphrase the provided information in your own words
                2. What does this information remind you of? Briefly explain the connection.
                3. Formulate two insightful questions that are raised by the text
                4. What can you deduce from the given information?"""
            ]
        }
        
        # MuSR-specific CEP prompts
        self.musr_ceps = {
            "murder_mystery": {
                "understand": [
                    "Summarize the key facts and relationships in this murder mystery story",
                    "Identify the main characters, their roles, and the central conflict"
                ],
                "connect": [
                    "What connections can you draw between the characters and events?",
                    "How do the different pieces of evidence relate to each other?"
                ],
                "query": [
                    "What questions does this story raise about motives and opportunities?",
                    "What aspects of the case seem most suspicious or unclear?"
                ],
                "application": [
                    "What conclusions can you draw about who is most likely responsible?",
                    "Based on the evidence, what are the strongest arguments for each suspect?"
                ],
                "comprehensive": [
                    """Please elaborate on this murder mystery by:
                    1. Summarize the key facts and identify the main characters
                    2. What connections can you draw between the characters and events?
                    3. What questions does this story raise about motives and opportunities?
                    4. What conclusions can you draw about who is most likely responsible?"""
                ]
            },
            "object_placements": {
                "understand": [
                    "Summarize the key events and object movements in this story",
                    "Identify the characters, objects, and their initial locations"
                ],
                "connect": [
                    "How do the different character perspectives relate to each other?",
                    "What connections can you draw between the events and character beliefs?"
                ],
                "query": [
                    "What questions does this story raise about what each character knows?",
                    "What aspects of the object movements seem most important?"
                ],
                "application": [
                    "What conclusions can you draw about where characters think objects are?",
                    "Based on the story, what are the most likely locations for each object?"
                ],
                "comprehensive": [
                    """Please elaborate on this object placement story by:
                    1. Summarize the key events and identify the objects and characters
                    2. How do the different character perspectives relate to each other?
                    3. What questions does this story raise about what each character knows?
                    4. What conclusions can you draw about where characters think objects are?"""
                ]
            },
            "team_allocation": {
                "understand": [
                    "Summarize the team allocation problem and the available people",
                    "Identify the tasks, people, and their skill levels"
                ],
                "connect": [
                    "How do the different skills and teamwork dynamics relate to each other?",
                    "What connections can you draw between individual abilities and team effectiveness?"
                ],
                "query": [
                    "What questions does this problem raise about optimal team composition?",
                    "What aspects of the allocation seem most challenging or important?"
                ],
                "application": [
                    "What conclusions can you draw about the best team assignments?",
                    "Based on the information, what are the strongest arguments for each allocation?"
                ],
                "comprehensive": [
                    """Please elaborate on this team allocation problem by:
                    1. Summarize the problem and identify the tasks, people, and skills
                    2. How do the different skills and teamwork dynamics relate to each other?
                    3. What questions does this problem raise about optimal team composition?
                    4. What conclusions can you draw about the best team assignments?"""
                ]
            }
        }
        
        # Baseline prompts
        self.baseline_prompts = {
            "direct": "Answer the following question based on the provided context:",
            "cot": "Let's approach this step by step. First, let me understand the question and the context, then I'll work through the reasoning to find the answer.",
            "cot_explicit": """Let's solve this step by step:

1. First, let me understand what the question is asking
2. Then, I'll identify the relevant information from the context
3. I'll trace through the reasoning needed to connect the information
4. Finally, I'll provide the answer

Question: {question}

Context:
{context}

Let me work through this:"""
        }
    
    def get_cep_prompt(self, category: str, prompt_index: int = 0, domain: str = "general") -> str:
        """
        Get a specific CEP prompt
        
        Args:
            category: The CEP category (understand, connect, query, application, comprehensive)
            prompt_index: Index of the prompt within the category
            domain: The domain (general, murder_mystery, object_placements, team_allocation)
            
        Returns:
            The CEP prompt string
        """
        if domain == "general":
            prompts = self.general_ceps.get(category, [])
        else:
            prompts = self.musr_ceps.get(domain, {}).get(category, [])
        
        if not prompts:
            raise ValueError(f"No prompts found for category '{category}' in domain '{domain}'")
        
        if prompt_index >= len(prompts):
            raise ValueError(f"Prompt index {prompt_index} out of range for category '{category}'")
        
        return prompts[prompt_index]
    
    def get_all_ceps(self, domain: str = "general") -> Dict[str, List[str]]:
        """
        Get all CEP prompts for a domain
        
        Args:
            domain: The domain to get prompts for
            
        Returns:
            Dictionary of category -> list of prompts
        """
        if domain == "general":
            return self.general_ceps
        else:
            return self.musr_ceps.get(domain, {})
    
    def get_baseline_prompt(self, prompt_type: str) -> str:
        """
        Get a baseline prompt
        
        Args:
            prompt_type: Type of baseline prompt (direct, cot, cot_explicit)
            
        Returns:
            The baseline prompt string
        """
        return self.baseline_prompts.get(prompt_type, "") 