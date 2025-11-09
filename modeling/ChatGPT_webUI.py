import openai
from revChatGPT.V1 import Chatbot

def prompt_chatgpt_and_parse_condition_response(
    chatbot,
    prompt,
    condition="precondition",
    method="chatgpt_api",
    openai_key=None,
):
    response = None
    if method == "chatgpt_web":
        for data in chatbot.ask(
          prompt
        ):
            response = data["message"]
    elif method == "chatgpt_api":
        openai.api_key = openai_key
        completion = chatbot.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt,
            }]
        )
        response = completion["choices"][0]["message"]["content"] 

    print("Prompt:", prompt)
    print("ChatGPT:", response)
    responses = response.split("\n")

    parsed_conditions = []
    for r in responses:
        label = None
        for i in range(1, 6):
            if "{}. ".format(i) in r:
                label =  "{}. ".format(i)
                break
            pass
        if label is not None:
            parsed_condition = r.split(label)[-1].strip()
            parsed_conditions.append(parsed_condition)
        pass
    return parsed_conditions


chatbot = Chatbot(config={
    "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJlZ2dub21lMDQxMUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZ2VvaXBfY291bnRyeSI6IlVTIn0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJ1c2VyX2lkIjoidXNlci1lT2hqZmdKcUVxZmxiVHNjNFhPVGxLTnEifSwiaXNzIjoiaHR0cHM6Ly9hdXRoMC5vcGVuYWkuY29tLyIsInN1YiI6Imdvb2dsZS1vYXV0aDJ8MTE1NjA3Mjg5ODU1MTc3MDc3NzQ1IiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY3NzU2MDUyMywiZXhwIjoxNjc4NzcwMTIzLCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9mZmxpbmVfYWNjZXNzIn0.o322d5-77gmofVGWrE_2e_M_TPBvdQhIPFWEoLAIXApaFAP4tC3_GQpm45p7AUe42tiuajkAcNVYRilQPNU6DHyvKTl9uDT4L8K5CNvO0roZgnPfZrHig1NUuUidIX2RitGQTG9dbFmmPm2v8x6zTJZjGRzCSVMYD8TAzgIaOwo62EYu2bg8BYO1X-OgoeRknAaMBVg3KXalVC6Ny5OabnTXhYgaj69Hy-5fLZiwQUDxz2iXWidJPBV6JTx3TZ83enOCp-way7HRex9i5ngKPR2zAQYKOf1CD5mKMgwH7rvP4MP2Abc5asMnSDgwnK3Z-9vC1a-9FKWarZNlmhZMIw"
})


if __name__ == "__main__":
    condition = "postcondition"
    prompt = "List three {}s of \"C turns the lawn mower\"?".format(condition)
    parsed_conditions = prompt_chatgpt_and_parse_condition_response(
        chatbot,
        prompt,
        condition=condition,
    )

    print(parsed_conditions)
