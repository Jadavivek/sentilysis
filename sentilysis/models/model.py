import pandas as pd 

from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

class EmotionResult(BaseModel):
    joy: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    neutral: float

    def wellness_score(self) -> float:
        positive = self.joy * 1.8 + self.neutral * 0.5
        negative = self.sadness * 0.8 + self.anger * 0.9 + self.fear * 0.9 + self.surprise * 0.5
        raw_score = positive - negative
        normalized_score = max(min(raw_score, 1.0), 0.0)
        return round(normalized_score, 3)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

parser = PydanticOutputParser(pydantic_object=EmotionResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an emotion detection AI. Return scores between 0 and 1 for each emotion."),
    ("user", "Tweet: {text}\n\n{format_instructions}")
])

chain: Runnable = prompt | llm | parser

def analyze_emotions_with_llm(tweets):
    results = []
    for tweet in tweets:
        try:
            structured = chain.invoke({"text": tweet['text'], "format_instructions": parser.get_format_instructions()})
            results.append(structured.model_dump())
        except Exception:
            results.append({e: 0.0 for e in EmotionResult.model_fields})
    
    print("results: ", results)
    
    df = pd.DataFrame(results)
    df['text'] = [tweet['text'] for tweet in tweets]
    df['timestamp'] = pd.to_datetime([tweet['timestamp'] for tweet in tweets], errors='coerce')
    df['date'] = df['timestamp'].dt.date
    return df
