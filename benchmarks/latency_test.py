import json, time, requests
from test_helper import API, get_token

token = get_token()
headers = {"Authorization": f"Bearer {token}"}

chars = requests.get(f"{API}/characters", headers=headers).json()
character_id = chars[0]["id"]

def time_one_message(chat_id, text):
    start = time.time()
    first_token = None
    r = requests.post(f"{API}/chats/{chat_id}/messages",
                      json={"content": text}, headers=headers, stream=True)
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        piece = json.loads(data)
        if "text" in piece and first_token is None:
            first_token = time.time() - start
    full = time.time() - start
    return first_token, full

chat_id = requests.post(f"{API}/chats", json={"character_id": character_id},
                        headers=headers).json()["chat_id"]

prompts = [
    "Hi, how are you?", "What are you reading lately?", "Tell me about your day.",
    "What's your favourite season?", "Do you like rainy days?",
    "What did you have for breakfast?", "Any book you'd recommend?",
    "What's the weather like there?", "Tell me something funny.",
    "What do you do on weekends?", "Do you like music?",
    "What's your favourite food?", "Have you travelled anywhere?",
    "What makes you happy?", "Do you have any hobbies?",
    "What's a perfect evening for you?", "Tell me a secret.",
    "What are you thinking about?", "Do you like tea or coffee?",
    "What should we do tomorrow?",
]

first_tokens, fulls = [], []
for text in prompts:
    ft, full = time_one_message(chat_id, text)
    if ft is None:
        print("skipped (model error)")
        continue
    first_tokens.append(ft)
    fulls.append(full)
    print(f"first token: {ft:.2f}s | full reply: {full:.2f}s")
    time.sleep(30)
print(f"\nMessages measured: {len(fulls)}")
print(f"AVERAGE first token: {sum(first_tokens)/len(first_tokens):.2f}s")
print(f"AVERAGE full reply:  {sum(fulls)/len(fulls):.2f}s")