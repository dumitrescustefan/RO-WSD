import os, sys, json, copy
import xlsxwriter
import stanza
import rowordnet as rwn

###### TEST 
from rowordnet import Synset
from tqdm.autonotebook import tqdm as tqdm

#stanza.download("ro")
wn = rwn.RoWordNet()

def read_and_annotate_sentences(file):
    with open(file, "r", encoding="utf8") as f:
        lines = f.readlines()

    nlp = stanza.Pipeline('ro')

    sentences = []
    for line in tqdm(lines, unit="sent"):
        sentence = {}
        sentence["text"] = line.strip()
        sentence["words"] = []
        sentence["lemmas"] = []
        sentence["pos"] = []
        sentence["span"] = []
        doc = nlp(line.strip())
        for word in doc.sentences[0].words:
            sentence["words"].append(word.text)
            sentence["lemmas"].append(word.lemma)
            sentence["pos"].append(word.upos)
            sentence["span"].append((word.parent.start_char,word.parent.end_char))
        sentences.append(sentence)
    return sentences

def run_wordnet(sentences):
    for i in range(len(sentences)):
        print("Processing [{}]".format(sentences[i]["text"]))
        sentences[i]["synsets"] = []
        for j in range(len(sentences[i]["words"])):
            word = sentences[i]["words"][j]
            lemma = sentences[i]["lemmas"][j]
            pos = sentences[i]["pos"][j]
            span = sentences[i]["span"][j]
            synsets = []

            if pos == "NOUN":
                synset_ids = wn.synsets(literal=lemma, strict=True, pos=Synset.Pos.NOUN)
                for synset_id in synset_ids:
                    s = wn(synset_id)
                    synset={}
                    synset["definition"] = s.definition
                    synset["literals"] = s.literals
                    synset["literals_senses"] = s.literals_senses
                    synsets.append(synset)
            sentences[i]["synsets"].append(synsets)

    return sentences



def generate_excel(sentences):

    def copy_format(book, fmt):
        properties = [f[4:] for f in dir(fmt) if f[0:4] == 'set_']
        dft_fmt = book.add_format()
        return book.add_format({k: v for k, v in fmt.__dict__.items() if k in properties and dft_fmt.__dict__[k] != v})

    filename = "wsd.xlsx"
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    header_row_format = workbook.add_format()
    header_row_format.set_bottom(1)
    header_row_format.set_bottom_color('#666666')
    header_row_format.set_bg_color('#CCCCCC')
    header_row_format.set_bold(True)

    # row formats
    row_format = workbook.add_format()
    row_format.set_bottom(1)
    row_format.set_bottom_color('#aaaaaa')
    row_format.set_bg_color('#ffffff')
    row_format.set_align('vcenter')

    alt_row_format = copy_format(workbook, row_format)
    alt_row_format.set_bg_color('#f0faff')

    # cell formats
    cell_format = workbook.add_format()
    cell_format.set_right(1)
    cell_format.set_right_color('#dddddd')
    cell_format.set_bottom(1)
    cell_format.set_bottom_color('#aaaaaa')
    cell_format.set_align('vcenter')

    alt_cell_format = copy_format(workbook, cell_format)
    alt_cell_format.set_bg_color('#f0faff')

    en_cell_format = copy_format(workbook, cell_format)
    en_cell_format.set_shrink()

    en_alt_cell_format = copy_format(workbook, alt_cell_format)
    en_alt_cell_format.set_shrink()

    cell_format.set_right(0)
    alt_cell_format.set_right(0)

    red_format = workbook.add_format({'color': 'red'})
    center_format = workbook.add_format({'align': 'center'})

    # Write some strings with multiple formats.


    """
    worksheet.set_default_row(25)
    worksheet.set_column('A:A', 6)
    worksheet.set_column('B:B', 2)
    worksheet.set_column('C:C', 100)
    worksheet.set_column('D:D', 60)

    worksheet.write(0, 0, "#")
    worksheet.write(0, 2, "Propozitie originala")
    worksheet.write(0, 3, "Traducere")
    worksheet.set_row(0, None, cell_format = header_row_format)
    """
    row = 0
    for sentence in sentences:
        for i in range(len(sentence["words"])):
            if sentence["synsets"][i]: # target word
                row+=1
                s, e = sentence["span"][i]
                text = sentence["text"]
                worksheet.write_rich_string(row, 0, text[0:s-1]+" ", red_format, text[s:e], text[e:]); row+=1
                for synset in sentence["synsets"][i]:
                    l = []
                    for ii in range(len(synset["literals"])):
                        l.append("{} - {}".format(synset["literals"][ii], synset["literals_senses"][ii]))
                    worksheet.write(row, 2, ", ".join(l)); row+=1
                    worksheet.write(row, 2, synset["definition"]); row += 1
                    worksheet.merge_range(row-2,1, row-1,1, "")
                #worksheet.write(row, 0, sentence["words"][i]); row+=1

            #worksheet.write_rich_string(row, 0,'This is ',red_format, 'bold',' and this is ')

        #worksheet.write(row, 0, sentence["text"])



        row += 1
        """
        for i in range(len(sentence["words"])):
            print("{}".format(sentence["words"][i]))
            if sentence["synsets"][i]:
                for synset_dict in sentence["synsets"][i]:
                    # print("\tsynset:")
                    print("\t\t {}".format(synset_dict["definition"]))
        """
    """
    row = 1
    for index in range(start, end):
        worksheet.write(row, 0, index)
        if index % 2 == 0:
            worksheet.set_row(row, None, cell_format=alt_row_format)
            worksheet.write(row, 2, sentence2id[index], en_alt_cell_format)
            worksheet.write(row, 3, '', alt_cell_format)
        else:
            worksheet.set_row(row, None, cell_format=row_format)
            worksheet.write(row, 2, sentence2id[index], en_cell_format)
            worksheet.write(row, 3, '', cell_format)

        row += 1
    """
    workbook.close()


sentences = read_and_annotate_sentences("sent.txt")
sentences = run_wordnet(sentences)
generate_excel(sentences)


