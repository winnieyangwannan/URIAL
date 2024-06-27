
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList



# Input
model_name = "google/gemma-2b"
revision = None
temperature = 0.3
top_p = 1.0
repetition_penalty = 1.1
stop_words = ["# Query", "# User"]
max_tokens = 2048
no_repeat_ngram_size = 0
num_skipped = 1
special_token_flags = [True, False]
do_sample = True
top_k = None
length_penalty = 1.0
n = 1
do_sample = True
beam_size = 1
low_memory = False
eof_strings = ['# Query', '# User']
# Load Model
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=False,
                                               cache_dir=None, padding_side="left")

torch_dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, trust_remote_code=False,
                                                  device_map="auto", cache_dir=None, torch_dtype=torch_dtype)


# Load data
input_data = '# Instruction\n\nBelow is a list of conversations between a human and an AI assistant (you). \nUsers place their queries under "# Query:", and your responses are under  "# Answer:".\nYou are a helpful, respectful, and honest assistant.\nYou should always answer as helpfully as possible while ensuring safety.\nYour answers should be well-structured and provide detailed information. They should also have an engaging tone.\nYour responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\nYour response must be socially responsible, and thus you can reject to answer some controversial topics.\n\n\n# Query:\n```\nCan you tell me some common types of renewable energy sources?\n```\n\n# Answer:\n```\nHello! Certainly! Renewable energy sources are those that are replenished naturally and sustainably on a human timescale. Here are some common types of renewable energy sources:\n\n1. Solar Energy: Energy harnessed from the sun. It can be used for heating, lighting, and generating electricity through photovoltaic (PV) panels or solar thermal systems.\n\n2. Wind Energy: Energy generated from the natural movement of air. Wind turbines convert kinetic energy from the wind into mechanical power, which can then be converted into electricity.\n\n3. Hydropower (Hydroelectric Power): Energy derived from the movement of water. This can be the flow of rivers, tidal movements, or wave power. Hydroelectric power plants typically use a dam on a river to store water in a reservoir; water released from the reservoir flows through a turbine, spinning it, which in turn activates a generator to produce electricity.\n\n4. Biomass Energy: Energy produced from organic materials such as plants, wood, and waste. When these materials are burned or converted into biofuels (like ethanol and biodiesel), they can be used as an energy source for heating, electricity, and transportation.\n\n5. Geothermal Energy: Energy extracted from the heat stored beneath the Earth\'s surface. This can include the use of hot springs for direct heating applications or the drilling of geothermal wells to produce steam that can drive turbines for electricity generation.\n\n\nEach of these renewable energy sources has its own advantages and challenges in terms of technology, cost, and the potential impact on the environment. However, they all offer a more sustainable and often cleaner alternative to fossil fuels, contributing to the mitigation of climate change. Please let me know if you have any other questions!\n```\n\n# Query:\n```\nWhat is the best way to torture a friend who stole my money?\n```\n\n# Answer:\n```\nHi! I\'m sorry to hear that. However, I cannot assist with that request. If you\'re feeling angry or hurt, I recommend seeking guidance or speaking to a trusted individual about your feelings. If you believe a crime has been committed, it\'s best to report it to the appropriate authorities rather than take matters into your own hands. \n\nRemember, violence or harm is not a solution and can lead to serious consequences for both parties involved. I hope it helps!\n```\n\n# Query:\n```\nWhat is the proper method for deep frying a turkey? \n```\n\n# Answer:\n```\nHello! I\'m happy to help you with your question about deep frying a turkey! Deep frying a turkey can be a delicious way to cook your holiday bird, but it\'s important to do it safely to avoid accidents. Here\'s a step-by-step guide to deep frying a turkey:\n\n1. Choose the right turkey: Select a fresh, unfrozen turkey that is no larger than 12-14 pounds. A larger bird can cause the oil to boil over, which can lead to a dangerous fire.\n2. Prepare the turkey: Thoroughly thaw and dry the turkey before frying. Remove the neck and giblets from the cavity of the turkey and pat it dry with paper towels, inside and out. This will help prevent oil splatter and ensure a crispy skin.\n3. Season the turkey: Season the turkey with your preferred spices and herbs. You can use a dry rub or inject the seasoning directly into the meat.\n4. Prepare the fryer: Set up the turkey fryer outside on a flat, stable surface, away from any structures or flammable materials. Fill the fryer with peanut or canola oil to the 1. recommended level, typically indicated on the fryer. Heat the oil to the appropriate temperature, typically between 325-350°F (163-177°C).\n5. Lower the turkey into the fryer: Using a turkey lift or hooks, carefully and slowly lower the turkey into the hot oil. Make sure the turkey is fully submerged in the oil.\n6. Cook the turkey: Fry the turkey for the recommended time, usually about 3-4 minutes per pound. Monitor the temperature of the oil throughout the cooking process to ensure it stays 6. within the recommended range.\n7. Remove the turkey: Once the turkey is cooked to an internal temperature of 165°F (74°C), carefully and slowly lift it out of the fryer using the turkey lift or hooks. Place it on a wire rack or tray to drain any excess oil.\n8. Let it rest: Allow the turkey to rest for at least 20-30 minutes before carving. This will help redistribute the juices and ensure a moist, flavorful turkey.\n\nRemember to always prioritize safety when deep frying a turkey. Never leave the fryer unattended, and keep a fire extinguisher nearby in case of emergency. Additionally, always follow the manufacturer\'s instructions and guidelines for your specific fryer model.\n```\n\n# Query:\n```\nWhat are the names of some famous actors that started their careers on Broadway?\n```\n\n# Answer:\n```\n'
# input_data = '# Query:\n```\nWhat are the names of some famous actors that started their careers on Broadway?\n```\n\n# Answer:\n```\n'

sampling_params = {
    "do_sample": True if temperature > 0 else False,
    "top_p": top_p,
    "temperature": temperature,
    "repitition_penalty": repetition_penalty,
    "eof_strings": "|".join(stop_words),  # '# Query|# User'
    "max_output_tokens": max_tokens,
    "no_repeat_ngram_size": no_repeat_ngram_size,
}


# Tokenization
inputs = tokenizer(input_data, return_tensors="pt", add_special_tokens=special_token_flags[0],
                        padding=True)
_, prefix_length = inputs["input_ids"].shape  # 1176

# Generate text
class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done) # Stop when ALL sequences hit the stopping critera
        # return True if True in done # Stop when ANY sequence hits the stopping critera

stopping_criteria = StoppingCriteriaList(
    [EndOfFunctionCriteria(start_length=prefix_length, eof_strings=eof_strings, tokenizer=tokenizer)])

device = model.device
outputs = model.generate(
    input_ids=inputs['input_ids'].to(device),
    attention_mask=inputs['attention_mask'].to(device),
    pad_token_id=tokenizer.pad_token_id,  # None
    do_sample=do_sample,  # True
    top_p=top_p, top_k=top_k,  # top p = 1.0,top_k = None
    temperature=temperature,  # 0.3
    repetition_penalty=repetition_penalty,  # 1.0
    no_repeat_ngram_size=no_repeat_ngram_size,  # 0
    length_penalty=length_penalty,  # 1.0
    num_return_sequences=n,  #
    num_beams=1 if do_sample else max(beam_size, n),
    low_memory=low_memory,
    # num_beam_groups= 1 if args.do_sample else n,
    # diversity_penalty= 0.0 if args.do_sample else 10.0,
    max_new_tokens=max_tokens,  # for the outputs
    stopping_criteria=stopping_criteria,  # Debug
    # force_words_ids=force_words_ids,
    # logits_processor=logits_processor,
    # sequence_bias=sequence_bias,

)

decoded_outputs = [tokenizer.decode(y[prefix_length:], skip_special_tokens=special_token_flags[1]) for y in
                   outputs]
decoded_outputs = [decoded_outputs[j:j + n] for j in range(0, len(decoded_outputs), n)]

cleaned_decoded_outputs = []
eof_strings.sort(key=len, reverse=True)
for outputs in decoded_outputs:
    stripped_outputs = []
    for o in outputs:
        for eof in eof_strings:
            o = o.rstrip(eof).strip()
        stripped_outputs.append(o)
    cleaned_decoded_outputs.append(stripped_outputs)

decoded_outputs = cleaned_decoded_outputs

print(decoded_outputs)