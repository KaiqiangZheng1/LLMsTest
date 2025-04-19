import time
from openai import OpenAI

client = OpenAI(api_key = 'xxx')

class LLM:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.client = client


    def llm_generate(
        self, 
        engine, 
        instruction,
        frames,
        system_info="You are a helpful AI assistant.",
        temp=0.8,
        top_p=1.0,
        max_tokens=2048
        ):

        cur_time = time.time()

        # Call the OpenAI API to get the softened sentence
        completion = self.client.chat.completions.create(
            model=engine,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=top_p,
            # Define the messages for the Chat Completion
            messages = [
                {
                    "role": "system",
                    "content": system_info
                },
                {
                    "role": "user",
                    "content": [instruction,
                                'The following are the metrics_pictures',
                                 *map(
                                    lambda x: {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{x}",
                                        },
                                    },
                                    frames,
                                )
                            ]
                }
            ]
        )
        # Extract the generated softened sentence
        response = completion.choices[0].message.content.strip()
        
        self.logger.info(f"Infer time: {time.time() - cur_time}s")

        return response




    def llm_generate_text_input(
        self, 
        engine, 
        instruction,
        system_info="You are a helpful AI assistant.",
        temp=0.8,
        top_p=1.0,
        max_tokens=2048
        ):

        cur_time = time.time()

        # Call the OpenAI API to get the softened sentence
        completion = self.client.chat.completions.create(
            model=engine,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=top_p,
            # Define the messages for the Chat Completion
            messages = [
                {
                    "role": "system",
                    "content": system_info
                },
                {
                    "role": "user",
                    "content": instruction,
                }
            ]
        )
        # Extract the generated softened sentence
        response = completion.choices[0].message.content.strip()
        
        self.logger.info(f"Infer time: {time.time() - cur_time}s")

        return response
