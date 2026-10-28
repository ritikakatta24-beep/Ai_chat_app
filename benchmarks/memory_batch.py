import json, time, requests
from test_helper import API, get_token

token = get_token()
headers = {"Authorization": f"Bearer {token}"}
character_id = requests.get(f"{API}/characters", headers=headers).json()[0]["id"]

# (what I say, what I ask later, word that must appear in the answer)
facts = [
    ("My sister's name is Meera.", "What is my sister's name?", "meera"),
    ("My favourite colour is teal.", "What is my favourite colour?", "teal"),
    ("I was born in Jaipur.", "Which city was I born in?", "jaipur"),
    ("My cat's name is Pickle.", "What is my cat's name?", "pickle"),
    ("My favourite movie is Interstellar.", "What is my favourite movie?", "interstellar"),
    ("I promised to bring you saffron tea next week.", "What did I promise to bring you?", "saffron"),
    ("My best friend is called Tanya.", "What is my best friend's name?", "tanya"),
    ("I play the violin.", "Which instrument do I play?", "violin"),
    ("My lucky number is 27.", "What is my lucky number?", "27"),
    ("My exam is in March.", "In which month is my exam?", "march"),
]

fillers = ["Do you like rainy days?", "What's your favourite book?",
           "Tell me something funny.", "Do you like tea or coffee?"]

def send(chat_id, text):
    r = requests.post(f"{API}/chats/{chat_id}/messages",
                      json={"content": text}, headers=headers, stream=True)
    reply = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        piece = json.loads(data)
        if "error" in piece:
            print("SERVER ERROR:", piece["error"])
        reply += piece.get("text", "")
    time.sleep(12)
    return reply

passed = 0
for statement, question, keyword in facts:
    chat_id = requests.post(f"{API}/chats", json={"character_id": character_id},
                            headers=headers).json()["chat_id"]
    send(chat_id, statement)
    for f in fillers:
        send(chat_id, f)
    answer = send(chat_id, question)
    ok = keyword in answer.lower()
    passed += ok
    print("PASS" if ok else "FAIL", "|", question, "->", answer[:100].replace("\n", " "))

print(f"\nMemory recall: {passed}/{len(facts)}")