#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope[openai, google]",
#     "pydantic",
#     "pygentic @ git+https://github.com/Bullish-Design/PyGentic.git"
# ]
# ///

"""
PyGentic Demo - Self-generating Pydantic models powered by Mirascope.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from pygentic import GenModel, generated_property, GenField, LLMConfig


class BookAnalyst(GenModel):
    """You are an expert literary analyst. Provide thoughtful, detailed
    analysis of books based on available information. Use scholarly yet
    accessible language."""

    title: str
    author: str

    genre: str = GenField(prompt="Determine the primary genre of '{title}' by {author}")
    publication_year: int = GenField(
        prompt="Find the publication year for '{title}' by {author}"
    )
    themes: List[str] = GenField(
        prompt="Identify 3-4 major themes in '{title}'. Return as list."
    )
    significance: str = GenField(
        prompt="Explain the literary significance of '{title}' in 2-3 sentences"
    )

    @generated_property(provider="openai", model="gpt-4o-mini")
    def detailed_analysis(self) -> str:
        """Write a comprehensive 200-word analysis of {title} by {author},
        covering its {themes} and {significance}"""
        pass

    @generated_property()
    def similar_books(self) -> List[str]:
        """Recommend 5 books similar to {title}, considering its {genre}
        and {themes}. Return only titles as list."""
        pass


class ProductReview(GenModel):
    """You are a professional product reviewer. Provide balanced, informative
    reviews that help consumers make purchasing decisions."""

    product_name: str
    brand: str

    price: float = GenField(prompt="Find current price for {product_name} by {brand}")
    category: str = GenField(prompt="Determine product category for {product_name}")
    key_features: List[str] = GenField(
        prompt="List 4-5 key features of {product_name} by {brand}"
    )
    pros: List[str] = GenField(
        prompt="Based on {key_features}, list 3-4 main advantages of {product_name}"
    )
    cons: List[str] = GenField(
        prompt="Identify 2-3 potential drawbacks of {product_name} in {category}"
    )

    @generated_property(temperature=0.3)
    def overall_rating(self) -> float:
        """Based on {pros} and {cons}, provide an overall rating from 1-5 for {product_name}.
        Return only the number."""
        pass

    @generated_property()
    def buying_recommendation(self) -> str:
        """Given the {overall_rating} rating, {pros}, and {cons}, write a 2-sentence
        buying recommendation for {product_name}"""
        pass


class RhymeGroup(BaseModel):
    """A model to represent rhyming groups of words."""

    rhyme_ending: str
    words: List[str]

    def __str__(self):
        return f"{self.rhyme_ending}: {', '.join(self.words)}"


class SemanticGroup(BaseModel):
    """A model to represent a semantic group of words."""

    category: str
    words: List[str]

    def __str__(self):
        return f"{self.category}: {', '.join(self.words)}"


class PhoneticPattern(BaseModel):
    """A model to represent phonetic patterns in words."""

    pattern: str
    words: List[str]

    def __str__(self):
        return f"{self.pattern}: {', '.join(self.words)}"


class VocabularyAnalyzer(GenModel):
    """You are an expert reading specialist and linguist. You analyze
    vocabulary lists to identify patterns, group related words, and
    find educational connections that help young learners."""

    level: str
    grade: str
    lesson: str
    words: List[str]
    word_count: int

    # Auto-generated groupings
    rhyme_groups: List[RhymeGroup] = GenField(
        prompt="Group the words {words} by rhyming patterns. Return as a dictionary where keys are the rhyme endings (like 'at', 'ick', 'ow') and values are lists of words that rhyme. Only include groups with 2+ words."
    )

    semantic_groups: List[SemanticGroup] = GenField(
        prompt="""Group the words {words} by meaning/topic. Create logical 
        categories like 'animals', 'actions', 'colors', 'family', etc. 
        Return as dictionary with category names as keys and word lists 
        as values."""
    )

    phonetic_patterns: List[PhoneticPattern] = GenField(
        prompt="""Group the words {words} by phonetic patterns like consonant 
        blends, vowel sounds, word endings. Focus on patterns useful for 
        grade {grade} phonics instruction. Return as dictionary."""
    )

    # difficulty_assessment: str = GenField(
    #     prompt="""Assess the overall difficulty of this {word_count}-word
    #     vocabulary list for grade {grade}. Consider word length, phonetic
    #     complexity, and age-appropriateness. Provide 2-3 sentences."""
    # )

    @generated_property(temperature=0.3)
    def topic_suggestions(self) -> str:
        """Based on the {rhyme_groups}, {semantic_groups}, and
        {phonetic_patterns}, suggest 5 potential story topics
        for grade {grade} students learning these words."""
        pass

    @generated_property()
    def story_plot_suggestions(self, topic: str) -> str:
        """Given a topic like '{topic}' and the following vocab list: {words}

        Suggest a simple story plot that incorporates the vocabulary words. Focus on
        creating an engaging narrative for young readers at a grade {grade} reading level.
        """
        pass

    @classmethod
    def from_vocab_data(cls, vocab_data: Dict[str, Any]) -> VocabularyAnalyzer:
        """Create VocabularyAnalyzer from parsed JSONL vocabulary data."""
        model = cls(
            level=vocab_data["level"],
            grade=str(vocab_data["grade"]),
            lesson=str(vocab_data["lesson"]),
            words=vocab_data["words"],
            word_count=vocab_data["word_count"],
        )
        print(f"    VocabularyAnalyzer created for level {model.level}:")
        print(f"\n{model}\n")
        return model


class StoryAnalyzer(GenModel):
    """You are an expert children's literature specialist and educator.
    You analyze children's stories for educational value, narrative
    quality, and age-appropriateness. Use precise, professional judgment."""

    story_id: str
    story_text: str
    # timestamp: datetime
    target_grade: str = "K-2"  # Default for early readers

    # Story quality ratings (1-10 scale)
    story_structure_rating: int = GenField(
        prompt="""Rate the story structure of this children's story on a 
        scale of 1-10. Consider: clear beginning/middle/end, logical 
        progression, appropriate pacing for young readers. Story: {story_text}"""
    )

    logical_flow_rating: int = GenField(
        prompt="""Rate the logical flow of this story on a scale of 1-10. 
        Consider: coherent sequence of events, cause-and-effect relationships, 
        smooth transitions. Story: {story_text}"""
    )

    vocab_usage_rating: int = GenField(
        prompt="""Rate the vocabulary usage in this story on a scale of 1-10. 
        Consider: age-appropriate word choices, repetition for learning, 
        variety without overwhelming young readers. Story: {story_text}"""
    )

    child_message_rating: int = GenField(
        prompt="""Rate the appropriateness and value of this story's message 
        for children on a scale of 1-10. Consider: positive themes, moral 
        lessons, emotional impact, safety. Story: {story_text}"""
    )

    # Detailed analysis
    story_themes: List[str] = GenField(
        prompt="""Identify 2-4 main themes in this children's story. 
        Focus on themes relevant to young readers. Story: {story_text}"""
    )

    # vocabulary_level: str = GenField(
    #     prompt="""Assess the vocabulary level of this story. Is it appropriate
    #     for kindergarten, 1st grade, 2nd grade, or higher? Consider word
    #     complexity and sentence structure. Story: {story_text}"""
    # )

    @generated_property(temperature=0.4)
    def detailed_critique(self) -> str:
        """Provide a comprehensive 150-word critique of this story,
        considering the ratings: structure ({story_structure_rating}/10),
        flow ({logical_flow_rating}/10), vocabulary ({vocab_usage_rating}/10),
        and message ({child_message_rating}/10). Include specific
        improvement suggestions."""
        pass

    # @generated_property()
    # def educational_value(self) -> str:
    #     """Analyze the educational value of this story. What skills
    #     does it help develop? What learning opportunities does it provide?
    #     Consider the {story_themes} and {vocabulary_level}."""
    #     pass

    # @generated_property(temperature=0.2)
    # def overall_score(self) -> float:
    #     """Calculate an overall quality score (1-10) based on the individual
    #     ratings: structure {story_structure_rating}, flow {logical_flow_rating},
    #     vocabulary {vocab_usage_rating}, message {child_message_rating}.
    #     Return only the numeric score."""
    #     pass

    @generated_property()
    def improvement_suggestions(self) -> List[str]:
        """Based on the ratings and analysis, provide 3-5 specific,
        actionable suggestions for improving this story for young readers."""
        pass

    @classmethod
    def from_story_data(cls, story_line: str) -> StoryAnalyzer:
        """Create StoryAnalyzer from parsed JSONL story data line."""
        parts = story_line.strip().split(" | ", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid story data format: {story_line}")

        story_id = parts[0]
        timestamp_str = parts[1]
        story_text = parts[2]

        # Parse timestamp
        # timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        return cls(
            story_id=story_id,
            story_text=story_text,
            # timestamp=timestamp
        )


def demo_vocab_analysis():
    """Demo vocabulary analysis using PyGentic."""
    print("\n=== PyGentic Vocabulary Analysis ===")
    # Load vocabulary data from JSONL file
    vocab_data = {
        "level": "K-2",
        "grade": 1,
        "lesson": 1,
        "words": ["cat", "hat", "bat", "rat", "mat"],
        "word_count": 5,
    }
    # Create VocabularyAnalyzer instance
    # analyzer = VocabularyAnalyzer.from_vocab_data(vocab_data)
    analyzer = VocabularyAnalyzer(
        level=vocab_data["level"],
        grade=str(vocab_data["grade"]),
        lesson=str(vocab_data["lesson"]),
        words=vocab_data["words"],
        word_count=vocab_data["word_count"],
    )

    # Access generated properties
    print(f"\nRhyme groups: {analyzer.rhyme_groups}")
    print(f"\nSemantic groups: {analyzer.semantic_groups}")
    print(f"\nPhonetic patterns: {analyzer.phonetic_patterns}\n")
    # Generate topic suggestions
    print(f"\nSuggested topics: {analyzer.topic_suggestions}\n")
    # Generate story plot suggestion for a specific topic
    # topic = analyzer.topic_suggestions[0]
    # print(f"\nStory plot suggestion for topic '{topic}':")
    # print(analyzer.story_plot_suggestions(topic))


def demo_basic_usage():
    """Demo basic PyGentic usage."""
    print("\n=== PyGentic Basic Usage ===")

    # Set global config
    config = LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.7)
    BookAnalyst.set_llm_config(config)

    # Create instance - fields auto-populate
    book = BookAnalyst(title="The Peripheral", author="William Gibson")

    print(f"{book}\n")
    print(f"    Generated book details:")
    print(f"      Title: {book.title}")
    print(f"      Author: {book.author}")
    print(f"      Genre: {book.genre}")
    print(f"      Publication Year: {book.publication_year}")
    print(f"      Themes: {book.themes}")
    print(f"      Significance: {book.significance}\n")

    # Access generated properties (cached after first call)
    print(f"Analysis: {book.detailed_analysis}...\n")
    print(f"Similar books: {book.similar_books}\n")


def demo_persistence():
    """Demo output and rehydration."""
    print("\n=== PyGentic Persistence ===")

    # Create with output file
    product = ProductReview(
        product_name="iPhone 15 Pro", brand="Apple", output_file="product_review.jsonl"
    )

    print(f"Generated product review:\n{product}\n")

    # Access properties to cache them
    rating = product.overall_rating
    recommendation = product.buying_recommendation

    # Save to JSONL
    product.output("product_review.jsonl")
    print(f"Saved to product_review.jsonl\n")

    # Rehydrate without LLM calls
    rehydrated = ProductReview.from_jsonl("product_review.jsonl")
    print(f"Rehydrated rating: {rehydrated.overall_rating}\n")
    print(f"Rehydrated recommendation: {rehydrated.buying_recommendation}\n")

    # Clean up
    Path("product_review.jsonl").unlink(missing_ok=True)


def demo_dependency_tracking():
    """Demo dependency tracking in generated properties."""
    print("\n=== PyGentic Dependency Tracking ===")

    class DynamicAnalysis(GenModel):
        """You are a data analyst. Provide insights based on current metrics."""

        product_name: str
        current_price: float
        competitor_price: float

        @generated_property(depends_on=["current_price", "competitor_price"])
        def price_analysis(self) -> str:
            """Analyze pricing for {product_name}: our price ${current_price} vs competitor ${competitor_price}"""
            pass

        @generated_property()  # Auto-detects dependencies from template
        def recommendation(self) -> str:
            """Based on {price_analysis}, recommend pricing strategy for {product_name}"""
            pass

    analysis = DynamicAnalysis(
        product_name="Wireless Headphones",
        current_price=199.99,
        competitor_price=179.99,
    )

    print(f"Initial price analysis: {analysis.price_analysis}...")
    print(f"Initial recommendation: {analysis.recommendation}...\n")

    # Update competitor price - should invalidate dependent properties
    print("Updating competitor price to $220...\n")
    analysis.competitor_price = 220.0

    print(f"Updated price analysis: {analysis.price_analysis}...")
    print(f"Updated recommendation: {analysis.recommendation}...\n")


def demo_advanced_features():
    """Demo advanced PyGentic features."""
    print("\n=== PyGentic Advanced Features ===")

    class ResearchPaper(GenModel):
        """You are an academic researcher. Provide rigorous, well-sourced
        analysis of research topics with proper academic methodology."""

        topic: str
        field: str

        research_questions: List[str] = GenField(
            prompt="Generate 3 specific research questions about {topic} in {field}"
        )
        methodology: str = GenField(
            prompt="Suggest appropriate research methodology for studying {research_questions} in {field}"
        )

        @generated_property(temperature=0.2)
        def abstract(self) -> str:
            """Write a 150-word academic abstract for research on {topic} using {methodology}
            to address {research_questions}"""
            pass

    # Configure for academic use
    ResearchPaper.set_llm_config(
        LLMConfig(
            provider="google",
            model="gemini-2.0-flash",
            temperature=0.3,  # Call this way for OpenAI
            system_prompt="Focus on rigorous academic standards and methodology.",
        )
    )

    paper = ResearchPaper(
        topic="Economic and Societal upheval from the introduction of LLMs",
        field="Economics and Sociology",
    )

    print("\nResearch Paper:")
    print(f"  Topic: {paper.topic}")
    print(f"  Field: {paper.field}")
    print(f"  Questions: {paper.research_questions}")
    print(f"  Methodology: {paper.methodology}")
    print(f"  Abstract: {paper.abstract}")
    print()


def main():
    """Run PyGentic demos."""
    print("\nPyGentic - Self-Generating Pydantic Models\n")

    try:
        demo_basic_usage()
        # demo_persistence()
        # demo_dependency_tracking()
        # demo_advanced_features()

        demo_vocab_analysis()

        print("\nPyGentic demo completed successfully!\n")

    except Exception as e:
        print(f"Demo failed: \n\n{e}\n\n")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
