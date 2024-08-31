from mlx_lm import load, generate


class Discussion(object):

    def __init__(self, model = "mlx-community/Meta-Llama-3-8B-Instruct-4bit", moderator = None, brainstormer = None, opposition = None) -> None:
        self.moderator = moderator
        self.brainstormer = brainstormer
        self.opposition = opposition
        self.model, self.tokenizer = load(model, adapter_path = None, model_config = {})
    
    def discuss(self, studies):
        history = []
        temp = 0.1
        mod_start_role = "You are moderating this scientific discussion. Your goal is to find a new topic to research in this area. You are to present some current studies to two other people. The first is \
            a brainstormer and the second is a critic. Start by giving the brainstormer some instructions while presenting these papers."
        prompt = self.tokenizer.apply_chat_template([{'role': 'system', 'content': mod_start_role}, {'role': 'user', 'content': str(studies)}], tokenize=False, add_generation_prompt=True)
        response = generate(self.model, self.tokenizer, prompt=prompt, temp=temp, verbose=True)
        history.append({'moderator': response})
        mod_role = "Keep this discussion going in a productive direction. Your goal is to lead the two other members of this discussion to find a new topic to research."
        brain_role = "You are creative and open. Come up with some new ideas. Base your answer off on the Discussion History."
        opp_role = "You are conservative and safe. Try to be realistic and give critiques about some of the ideas presented. Base your answer on the Discussion History."
        roles = {'brainstormer': brain_role, 'moderator': mod_role, 'opposition': opp_role}
                 
        for x in range(5):
            for name, role in roles.items():
                input = "DISCUSSION HISTORY: " + str(history[-3:]) + "\n\n"
                prompt = self.tokenizer.apply_chat_template([{'role': 'system', 'content': role}, {'role': 'user', 'content': input}], tokenize=False, add_generation_prompt=True)
                response = generate(self.model, self.tokenizer, prompt=prompt, temp=temp, verbose=True)
                history.append({name: response})

