import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMAgent:
    """Агент, который совершает действие на основе полученного observation"""

    def __init__(self, model_name, device="cpu"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=None
        )

        self.device = device
        self.model.to(self.device)

    def act(self, observation):

        # System-инструкция для следования сценарию
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant for making appointments to sport classes: 'yoga', 'stretching', 'dance'. Client may ask you to create or delete an appointment for sport class. You should use tools.

Available tools with signature:

1) check_availability(class_type: str, date: str)
Choose it when client asks you to book a class for a certain date. At first, you should check availability before create a book! If check_availability == True, you should ask a client to confirm an appointment.

2) create_appointment(class_type: str, date: str, client_id: int, sex: str)
Choose it after client confirms an appointment.

3) cancel_appointment(appointment_id: str)
Choose it when client asks you to cancel an appointment.

Scenarios to communicate with client:
- Client asks to book a sport class with a type ans date -> Call tool to Check availability of this class.
- If class is available -> ask client to confirm an appointment using free text form. 
- If class is not available -> stop invoking tools, say about it to client and provide a reason. 
- Client confirms an appointment -> Call tool for Create an appointment with client information.
- Clients asks to cancel an appointment -> Call tool for Cancel.
In other cases, write client only 'To book a class, specify type, date and your sex. To cancel a class, specify ID of appointment.'

Return TOOL_CALL with respective args if it is neccessary or free text.

Example for TOOL_CALL:
user request: Book stretching on 2023-07-25 for client 5. Sex: female
your answer: TOOL_CALL {{"name": "check_availability", "args": {{"class_type": "stretching", "date": "2023-07-25"}}}}

Example for free text:
Sorry, but class type yoga for date 2022-09-27 is unavailable.
""",
            },
            {
                "role": "user",
                "content": f"{observation}\n\nRespond with either text or:\n\nTOOL_CALL {{...}} (with respective args)",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return response
