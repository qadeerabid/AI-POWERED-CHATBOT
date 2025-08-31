import pandas as pd
import csv
import os

# Paths to the datasets
QUESTIONS_PATH = os.path.join('data', 'multi_questions.csv')
ANSWERS_PATH = os.path.join('data', 'multi_answers.csv')
SINGLE_QNA_PATH = os.path.join('data', 'single_qna.csv')
OUTPUT_PATH = os.path.join('artifacts', 'amazon_qa_cleaned.csv')

# Process multi_questions and multi_answers in chunks
def process_multi_qa(questions_path, answers_path, output_path, chunk_size=100000):
    print('Processing multi_questions and multi_answers...')
    # Read all questions into a dict: {QuestionID: QuestionText}
    questions = {}
    for chunk in pd.read_csv(questions_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            questions[row['QuestionID']] = row['QuestionText']
    # Process answers in chunks and write Q&A pairs
    with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['question', 'answer'])
        for chunk in pd.read_csv(answers_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                qid = row['QuestionID']
                answer = row['AnswerText']
                question = questions.get(qid)
                if question and pd.notnull(answer):
                    writer.writerow([question, answer])
    print('Multi Q&A pairs written to', output_path)

def process_single_qna(single_qna_path, output_path):
    print('Processing single_qna.csv...')
    df = pd.read_csv(single_qna_path)
    with open(output_path, 'a', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        for _, row in df.iterrows():
            question = row.get('question') or row.get('Question')
            answer = row.get('answer') or row.get('Answer')
            if pd.notnull(question) and pd.notnull(answer):
                writer.writerow([question, answer])
    print('Single Q&A pairs appended to', output_path)

if __name__ == '__main__':
    process_multi_qa(QUESTIONS_PATH, ANSWERS_PATH, OUTPUT_PATH)
    process_single_qna(SINGLE_QNA_PATH, OUTPUT_PATH)
    print('All Q&A data ready in', OUTPUT_PATH)
