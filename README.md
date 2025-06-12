# PyGentic 🧬

**LLM Mad-libs, based on Pydantic**

Ever wished your Pydantic models could just... figure themselves out? PyGentic makes that dream reality. Define your data structure once, provide a few prompts, and watch as your models populate themselves using the power of LLMs.

Think of it as giving your data classes a brain. A very expensive, minimally intelligent, occasionally hallucinating brain, but a brain nonetheless.

## ✨ What Is This Sorcery?

PyGentic extends Pydantic with self-generating capabilities powered by [Mirascope](https://github.com/Mirascope/mirascope). Instead of manually populating every field, you write prompt templates and let LLMs do the heavy lifting.

```python
from pygentic import GenModel, GenField, generated_property

class BookReview(GenModel):
    """You are a professional book critic with impeccable taste."""
    
    title: str  # You provide this
    author: str  # And this
    
    # Everything else? Let the LLM figure it out
    genre: str = GenField(prompt="What genre is '{title}' by {author}?")
    rating: float = GenField(prompt="Rate '{title}' from 1-5")
    review: str = GenField(prompt="Write a witty review of '{title}'")
    
    @generated_property()
    def recommendation(self) -> str:
        """Based on the {rating} and {review}, would you recommend '{title}'?"""
        pass

# Magic happens here ✨
review = BookReview(title="Dune", author="Frank Herbert")
print(review.genre)        # "Science Fiction"
print(review.rating)       # 4.7
print(review.review)       # "A sprawling space opera that..."
print(review.recommendation)  # "Absolutely! If you enjoy..."
```

## 🚀 Features

- **Self-Populating Fields**: Define prompt templates, get auto-generated content
- **Dependency Tracking**: Properties regenerate when their inputs change
- **Type-Aware Generation**: Automatic structured outputs using Pydantic models
- **Caching**: LLM calls are cached until dependencies change
- **Persistence**: Save and restore fully-populated models without re-generation
- **Provider Agnostic**: Works with OpenAI, Anthropic, Google, and more via Mirascope
- **Pure Pydantic**: Familiar syntax, enhanced with LLM superpowers

## 📦 Installation

```bash
uv add pygentic
```

Wait. You're not using UV yet? You should get on that. But in the meantime, go ahead and try...

```bash
pip install pygentic
```

## 🎯 Quick Start

### Basic Usage
Think of this exactly as you'd think of Pydantic. Instead of BaseModel, import GenModel. Instead of Field, import GenField. Clever, no?
```python
from pygentic import GenModel, GenField

class ProductAnalysis(GenModel):
    """You are a market research expert with an eye for trends."""
    
    product_name: str
    company: str
    
    category: str = GenField(prompt="What category is {product_name}?")
    target_audience: str = GenField(prompt="Who is the target audience for {product_name}?")
    competitors: list[str] = GenField(prompt="List 3 main competitors to {product_name}")

analysis = ProductAnalysis(
    product_name="iPhone 15 Pro", 
    company="Apple"
)

# Fields auto-populate on creation
print(analysis.category)        # "Smartphone"
print(analysis.target_audience) # "Tech enthusiasts and professionals"
print(analysis.competitors)     # ["Samsung Galaxy S23", "Google Pixel 8", ...]
```

### Advanced: Generated Properties
Generated properties allow you to hold off on expensive LLM calls until they're needed. As such, they've got some handy ways to customize the API calls to use different models from the GenModel class. Once a generated_property is called, it stays cached. Unless you want them to update, then you can decide what attributes kick off the regeneration.  
```python
from pygentic import GenModel, GenField, generated_property

class MarketStrategy(GenModel):
    """You are a brilliant marketing strategist."""
    
    product: str
    budget: float
    timeline: str = GenField(prompt="Suggest launch timeline for {product}")
    
    @generated_property(provider="anthropic", temperature=0.8)
    def campaign_ideas(self) -> list[str]:
        """Generate 5 creative marketing campaign ideas for {product} 
        with budget ${budget} over {timeline}"""
        pass
    
    @generated_property(depends_on=["budget", "timeline"])
    def budget_breakdown(self) -> dict[str, float]:
        """Break down ${budget} budget across {timeline} timeline"""
        pass

strategy = MarketStrategy(product="Smart Water Bottle", budget=50000)
print(strategy.campaign_ideas)    # ["Hydration Station Pop-ups", ...]
print(strategy.budget_breakdown)  # {"digital_ads": 20000, "events": 15000, ...}

# Update budget - dependent properties auto-regenerate
strategy.budget = 75000
print(strategy.budget_breakdown)  # Updated with new budget!
```

### Persistence Magic
PyGentic is meant to be agentic. Right now, that doesn't mean too much, but the end goal is to have swarms of little models running around in the background taking care of things for you. To that end, models are able to save/load themselves for later reference. Right now that just means reading and writing from a jsonl log. Eventually, it'll be DB oriented. 
```python
# Save your expensive LLM-generated content
analysis.output("my_analysis.jsonl")

# Later, restore without any LLM calls
restored = ProductAnalysis.from_jsonl("my_analysis.jsonl")
print(restored.competitors)  # Instant access, no API calls
```

## 🎨 Configuration
Turns out you can really burn through some cash with these LLM providers. Because of this, PyGentic allows fine grained control over API provider/model settings. Each GenModel has a global API call (generally this should be a cheap/local model), but each property can call different models, with different settings. 

### Global Configuration

```python
from pygentic import LLMConfig

# Set up your preferred LLM
config = LLMConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-latest",
    temperature=0.7
)

# Apply to all instances of a class
ProductAnalysis.set_llm_config(config)
```

### Per-Property Configuration

```python
@generated_property(
    provider="openai",           # Override provider
    model="gpt-4o",             # Override model  
    temperature=0.9,            # Override temperature
    depends_on=["price", "features"]  # Explicit dependencies
)
def marketing_copy(self) -> str:
    """Write compelling copy for {product} at ${price}"""
    pass
```

## 🧠 How It Works

1. **Template Resolution**: `{field_name}` patterns in prompts get replaced with actual values
2. **Dependency Tracking**: Properties automatically detect dependencies from templates
3. **Smart Caching**: Results are cached until dependencies change
4. **Type Safety**: Return types become Mirascope response models for structured output
5. **Lazy Generation**: Fields populate only when accessed (unless required)

## 🎭 Real-World Examples

### Content Creation Pipeline

```python
class BlogPost(GenModel):
    """You are a skilled content creator and SEO expert."""
    
    topic: str
    target_audience: str
    
    title: str = GenField(prompt="Create an engaging title about {topic} for {target_audience}")
    outline: list[str] = GenField(prompt="Create a 5-point outline for '{title}'")
    keywords: list[str] = GenField(prompt="Suggest 10 SEO keywords for {topic}")
    
    @generated_property()
    def content(self) -> str:
        """Write a full blog post with title '{title}' following {outline}"""
        pass
    
    @generated_property()
    def meta_description(self) -> str:
        """Write SEO meta description for '{title}' using {keywords}"""
        pass

post = BlogPost(topic="sustainable gardening", target_audience="urban millennials")
# Entire content pipeline generates automatically
```

### Data Analysis Workflow

```python
class DataInsights(GenModel):
    """You are a senior data scientist with expertise in business analytics."""
    
    dataset_description: str
    business_context: str
    
    key_metrics: list[str] = GenField(
        prompt="Identify 5 key metrics to analyze for {dataset_description} in {business_context}"
    )
    
    @generated_property()
    def analysis_plan(self) -> dict[str, str]:
        """Create analysis plan for {key_metrics} in context of {business_context}"""
        pass
    
    @generated_property(temperature=0.3)
    def recommendations(self) -> list[str]:
        """Based on {analysis_plan}, provide 3 actionable business recommendations"""
        pass
```

## 🤝 Contributing

Found a bug? Have a feature idea? Contributions welcome!

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Add tests (if you're feeling fancy)
5. Submit a PR

## 📄 License

MIT + Commons Clause License

Free for personal and open-source use. Commercial use requires a license.

## 🙏 Acknowledgments

Built on the shoulders of giants:
- [Mirascope](https://github.com/Mirascope/mirascope) - The LLM Swiss Army knife
- [Pydantic](https://github.com/pydantic/pydantic) - The validation MVP

## 🎪 Why "PyGentic"?

Because your models are now genetically enhanced with LLM DNA. They've evolved beyond simple data containers into self-aware, self-populating entities. This is fine. 

In entirely unrelated news, did you know that the unemployment rate at the peak of the great depression was ~25%? It mostly varied between 10%-20% throughout. Those that managed to keep their jobs saw their income fall over 40% between 1929 and 1933. 

Thanks for checking out the library, feel free to leave any thoughts or ideas in the repo discussions. Enjoy automating your work away!

---

**Made with ❤️ and probably too much caffeine**
