from prompt2model.utils.information_extractor import InformationExtractor
info_extractor = InformationExtractor(pretrained_model_name="sshleifer/tiny-gpt2")
input_extraction_prompt = info_extractor.construct_input_prompt(
    instruction="Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.",
    input="""[input]='Question: What is the largest desert in the world?'

The answer to the question can vary depending on the definition used for "largest desert," but the largest deserts in the world are generally considered to be the Antarctic Desert, the Arctic Desert, the Gobi Desert, the Great Victoria Desert, and the Kalahari Desert.",
    """
)
output_extraction_prompt = info_extractor.construct_output_prompt(
    instruction="Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.",
    input="What is the capital city of China?",
    output="Answer:Beijing is the capital city of China."
)
