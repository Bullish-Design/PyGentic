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
from typing import List

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


def demo_basic_usage():
    """Demo basic PyGentic usage."""
    print("\n=== PyGentic Basic Usage ===")

    # Set global config
    config = LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.7)
    BookAnalyst.set_llm_config(config)

    # Create instance - fields auto-populate
    book = BookAnalyst(title="The Peripheral", author="William Gibson")

    print(f"{book}\n")

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
        demo_persistence()
        # demo_dependency_tracking()
        demo_advanced_features()

        print("\nPyGentic demo completed successfully!\n")

    except Exception as e:
        print(f"Demo failed: \n\n{e}\n\n")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
