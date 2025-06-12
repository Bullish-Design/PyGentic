#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope",
#     "pydantic",
# ]
# ///

"""
Demo of the self-generating Pydantic system.
"""

from __future__ import annotations

from typing import List
from pydantic import Field

from . import SelfGeneratingBase, generated_property, LLMConfig, GenField


class BookAnalyst(SelfGeneratingBase):
    """You are an expert literary analyst. Provide thoughtful, detailed 
    analysis of books based on available information. Use scholarly yet 
    accessible language."""
    
    title: str
    author: str
    
    genre: str = GenField(prompt="Determine the primary genre of '{title}' by {author}")
    
    publication_year: int = GenField(prompt="Find the publication year for '{title}' by {author}")
    
    themes: List[str] = GenField(prompt="Identify 3-4 major themes in '{title}'. Return as list.")
    
    significance: str = GenField(prompt="Explain the literary significance of '{title}' in 2-3 sentences")
    
    @generated_property(provider="anthropic", model="claude-3-5-sonnet-latest")
    def detailed_analysis(self) -> str:
        """Write a comprehensive 200-word analysis of {title} by {author}, 
        covering its {themes} and {significance}"""
        pass
    
    @generated_property()
    def similar_books(self) -> List[str]:
        """Recommend 5 books similar to {title}, considering its {genre} 
        and {themes}. Return only titles as list."""
        pass


class ProductReview(SelfGeneratingBase):
    """You are a professional product reviewer. Provide balanced, informative 
    reviews that help consumers make purchasing decisions."""
    
    product_name: str
    brand: str
    price: float = GenField(prompt="Find current price for {product_name} by {brand}")
    
    category: str = GenField(prompt="Determine product category for {product_name}")
    
    key_features: List[str] = GenField(prompt="List 4-5 key features of {product_name} by {brand}")
    
    pros: List[str] = GenField(prompt="Based on {key_features}, list 3-4 main advantages of {product_name}")
    
    cons: List[str] = GenField(prompt="Identify 2-3 potential drawbacks of {product_name} in {category}")
    
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


def demo_book_analyst():
    """Demo the BookAnalyst class."""
    print("=== BookAnalyst Demo ===")
    
    # Set default LLM config
    config = LLMConfig(
        provider="openai", 
        model="gpt-4o-mini",
        temperature=0.7
    )
    BookAnalyst.set_llm_config(config)
    
    # Create instance with minimal data
    book = BookAnalyst(title="1984", author="George Orwell")
    
    print(f"Title: {book.title}")
    print(f"Author: {book.author}")
    print(f"Genre: {book.genre}")
    print(f"Year: {book.publication_year}")
    print(f"Themes: {book.themes}")
    print(f"Significance: {book.significance}")
    print()
    
    # Access generated properties (cached after first call)
    print("Detailed Analysis:")
    print(book.detailed_analysis)
    print()
    
    print("Similar Books:")
    for i, title in enumerate(book.similar_books, 1):
        print(f"{i}. {title}")
    print()


def demo_product_review():
    """Demo the ProductReview class."""
    print("=== ProductReview Demo ===")
    
    # Create with partial data
    product = ProductReview(
        product_name="iPhone 15 Pro", 
        brand="Apple"
    )
    
    print(f"Product: {product.product_name}")
    print(f"Brand: {product.brand}")
    print(f"Price: ${product.price}")
    print(f"Category: {product.category}")
    print(f"Key Features: {product.key_features}")
    print(f"Pros: {product.pros}")
    print(f"Cons: {product.cons}")
    print()
    
    # Generated properties
    print(f"Rating: {product.overall_rating}/5")
    print(f"Recommendation: {product.buying_recommendation}")
    print()


def demo_custom_config():
    """Demo with custom LLM configurations."""
    print("=== Custom Config Demo ===")
    
    class MovieRecommender(SelfGeneratingBase):
        """You are a knowledgeable film critic and movie enthusiast. 
        Provide insightful recommendations based on user preferences."""
        
        favorite_genre: str
        favorite_actor: str
        
        mood: str = GenField(prompt="Based on {favorite_genre} preference, suggest what mood this person might be in when watching movies")
        
        recommendation: str = GenField(prompt="Recommend one specific movie featuring {favorite_actor} in {favorite_genre} genre for someone in a {mood} mood")
        
        @generated_property(provider="anthropic", temperature=0.9)
        def creative_pitch(self) -> str:
            """Write a creative, engaging 50-word pitch for why someone should watch {recommendation}"""
            pass
    
    # Set high-creativity config
    MovieRecommender.set_llm_config(LLMConfig(
        provider="openai",
        model="gpt-4o-mini", 
        temperature=0.8
    ))
    
    recommender = MovieRecommender(
        favorite_genre="sci-fi",
        favorite_actor="Ryan Gosling"
    )
    
    print(f"Genre: {recommender.favorite_genre}")
    print(f"Actor: {recommender.favorite_actor}")
    print(f"Mood: {recommender.mood}")
    print(f"Recommendation: {recommender.recommendation}")
    print(f"Pitch: {recommender.creative_pitch}")
    print()
    """Demo with custom LLM configurations."""
    print("=== Custom Config Demo ===")
    
    class MovieRecommender(SelfGeneratingBase):
        """You are a knowledgeable film critic and movie enthusiast. 
        Provide insightful recommendations based on user preferences."""
        
        favorite_genre: str
        favorite_actor: str
        
        mood: str = GenField(prompt="Based on {favorite_genre} preference, suggest what mood this person might be in when watching movies")
        
        recommendation: str = GenField(prompt="Recommend one specific movie featuring {favorite_actor} in {favorite_genre} genre for someone in a {mood} mood")
        
        @generated_property(provider="anthropic", temperature=0.9)
        def creative_pitch(self) -> str:
            """Write a creative, engaging 50-word pitch for why someone should watch {recommendation}"""
            pass
    
    # Set high-creativity config
    MovieRecommender.set_llm_config(LLMConfig(
        provider="openai",
        model="gpt-4o-mini", 
        temperature=0.8
    ))
    
    recommender = MovieRecommender(
        favorite_genre="sci-fi",
        favorite_actor="Ryan Gosling"
    )
    
    print(f"Genre: {recommender.favorite_genre}")
    print(f"Actor: {recommender.favorite_actor}")
    print(f"Mood: {recommender.mood}")
    print(f"Recommendation: {recommender.recommendation}")
    print(f"Pitch: {recommender.creative_pitch}")
    print()


def main():
    """Run all demos."""
    print("Self-Generating Pydantic Demo\n")
    
    try:
        demo_book_analyst()
        demo_product_review() 
        demo_output_and_rehydration()
        demo_custom_config()
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
