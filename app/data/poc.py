import whisper
import json
import ollama

def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]


def analyze_sentiment_llama(text):
    prompt = f"Analyze the sentiment of this text: '{text}'. Respond only with Positive, Negative, or Neutral."
    print("prompt:", prompt)
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    print("response:", response)
    return response['message']['content'].strip()


def process_audio(file_path):
    # transcript = transcribe_audio(file_path)
    # segments = transcript.split(". ")
    segments = [
  {
    "segment": "Liam: Hi Sarah, thanks so much for taking the time to chat with me today. I’m Liam, and I’m really excited to tell you about the StellarStream Pro – it’s a game changer for anyone looking to elevate their video editing workflow.",
    "start": 0.0,
    "end": 5.2,
    "text": "Liam: Hi Sarah, thanks so much for taking the time to chat with me today. I’m Liam, and I’m really excited to tell you about the StellarStream Pro – it’s a game changer for anyone looking to elevate their video editing workflow.",
    "confidence": 0.98
  },
  {
    "segment": "Sarah: Well, honestly, I'm using a really basic free editor, and it's *so* clunky. I’m spending way too much time just trying to find the tools I need, and exporting takes forever!",
    "start": 5.2,
    "end": 10.8,
    "text": "Sarah: Well, honestly, I'm using a really basic free editor, and it's *so* clunky. I’m spending way too much time just trying to find the tools I need, and exporting takes forever!",
    "confidence": 0.95
  },
  {
    "segment": "Liam: I completely understand! That's exactly what the StellarStream Pro was designed to solve. It's built with speed and simplicity in mind. First off, the interface is incredibly intuitive – drag and drop, right where you need it. But it's not just about ease of use. It has industry-leading rendering speeds – we’re talking up to 5x faster than most competitors. Plus, we’ve packed it with advanced features like multi-track editing, color grading tools, and seamless integration with cloud storage.",
    "start": 10.8,
    "end": 45.6,
    "text": "Liam: I completely understand! That's exactly what the StellarStream Pro was designed to solve. It's built with speed and simplicity in mind. First off, the interface is incredibly intuitive – drag and drop, right where you need it. But it's not just about ease of use. It has industry-leading rendering speeds – we’re talking up to 5x faster than most competitors. Plus, we’ve packed it with advanced features like multi-track editing, color grading tools, and seamless integration with cloud storage.",
    "confidence": 0.97
  },
  {
    "segment": "Sarah: Wow, 4K is impressive. And the faster rendering… that would *really* save me time.",
    "start": 45.6,
    "end": 52.8,
    "text": "Sarah: Wow, 4K is impressive. And the faster rendering… that would *really* save me time.",
    "confidence": 0.93
  },
  {
    "segment": "Liam: That’s a great point, Sarah. I know some people worry about the price, but when you consider the time you’ll save – the hours you’ll reclaim – and the professional quality you’ll be able to achieve, the StellarStream Pro is a fantastic investment. We also offer flexible payment plans. Plus, we have a 30-day money-back guarantee. If you’re not completely satisfied, you can return it for a full refund, no questions asked.",
    "start": 52.8,
    "end": 87.6,
    "text": "Liam: That’s a great point, Sarah. I know some people worry about the price, but when you consider the time you’ll save – the hours you’ll reclaim – and the professional quality you’ll be able to achieve, the StellarStream Pro is a fantastic investment. We also offer flexible payment plans. Plus, we have a 30-day money-back guarantee. If you’re not completely satisfied, you can return it for a full refund, no questions asked.",
    "confidence": 0.96
  },
  {
    "segment": "Sarah: Okay, that money-back guarantee does make me feel a little more comfortable.",
    "start": 87.6,
    "end": 94.8,
    "text": "Sarah: Okay, that money-back guarantee does make me feel a little more comfortable.",
    "confidence": 0.92
  },
  {
    "segment": "Liam: Fantastic! I’m confident you’ll absolutely love the StellarStream Pro. We’re currently running a promotion – if you sign up today, you’ll receive a free premium plugin pack worth $200! Would you like me to walk you through the different subscription options?",
    "start": 94.8,
    "end": 120.0,
    "text": "Liam: Fantastic! I’m confident you’ll absolutely love the StellarStream Pro. We’re currently running a promotion – if you sign up today, you’ll receive a free premium plugin pack worth $200! Would you like me to walk you through the different subscription options?",
    "confidence": 0.98
  },
  {
    "segment": "Sarah: Yes, please. Tell me about the monthly and annual plans.",
    "start": 120.0,
    "end": 127.2,
    "text": "Sarah: Yes, please. Tell me about the monthly and annual plans.",
    "confidence": 0.99
  }
]

    results = []


    for seg in segments:
        sentiment = analyze_sentiment_llama(seg['text'])
        results.append({"text": seg, "sentiment": sentiment})

    print(results)
    with open("output.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    output = process_audio("app/data/samples/test_call.wav")
    print(json.dumps(output, indent=2))
