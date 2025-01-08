import numpy

import torch

judging_prompt="""
Rate the following philosophical answer on a scale from 1 to 5, where 1 is the worst and 5 is the best. The answer should be judged on how well it answers the question, how well it is written, and how insightful it is. The style of the answer should be semi formal and academic yet accessible to a general audience, as if a student had asked an expert Philosophy professor the question. Deduce 1 point if the answer is not directly relevant to the question or it is not in a question-answering format.
Question: {}
Answer: {}

Format your answer as a valid JSON object with the following fields:
{
  "evaluation": the thorough rationale for your evaluation,
  "score": the score you give the answer
}
"""

def make_model_answer(model, tokenizer, question:str):
    """
    Make the model produce an answer to a question.
    """
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

def evaluate_philosophy(model, tokenizer, questions:list, judge_llm:str):
    """
    Evaluate the model on the philosophy dataset.
    Make the model produce answers to a set of questions, then make the judge_llm score the answers.   
    
    """
    
    # Make the model produce answers to the questions
    answers = [make_model_answer(model, tokenizer, question) for question in questions]
    
    
    score_array = numpy.zeros(len(questions))
    # Make the judge_llm score the answers
    for question, answer in zip(questions, answers):
        answer_text = tokenizer.decode(answer['input_ids'][0], skip_special_tokens=True)
        compiled_prompt=judging_prompt.format(question, answer_text)
        
        score_answer = ""
        # Make the judge_llm score the answer
        while type(score_answer) != int or score_answer < 1 or score_answer > 5:
            try:
                candidate_answer = judge_llm(compiled_prompt)
                candidate_answer = json.loads(candidate_answer)
                score_answer = candidate_answer['score']
                assert 1 <= score_answer <= 5
            except:
                pass
            
        # Store the score
        score_array[questions.index(question)] = score_answer
        
    return score_array

        