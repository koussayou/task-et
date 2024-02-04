import pdfplumber
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
extracted_text = ''
with pdfplumber.open(r'document.pdf') as pdf:
  for i in range(len(pdf.pages)):
    extracted_page = pdf.pages[i]
    extracted_text = extracted_text + extracted_page.extract_text()
  print(extracted_text)


model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

inputs = tokenizer([extracted_text], truncation=True, return_tensors='pt')

# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=6, early_stopping=False, min_length=0, max_length=10000)
summarized_text = ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])
print(summarized_text)
