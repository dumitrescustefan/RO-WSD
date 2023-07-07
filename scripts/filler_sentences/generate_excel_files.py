import json, os
import rowordnet as rwn
import xlsxwriter

_LEGEND_ = "       Introduceți numele dumneavoastră, apoi completați toate celulele albastre cu câte o propoziție care sa conțină cuvântul din celula roșie, între acolade { } și cu sensul definit de celula portocalie."
_MAXIMUM_FILE_COUNT_ = 10
_MINIMUM_SENTENCES_PER_FILE_ = 100
_MAX_LITERALS_ = 5000
_INPUT_PATH_ = "scripts/filler_sentences/input/"
_OUTPUT_PATH_ = "scripts/filler_sentences/output/"
_LOCK_ = False # This feature is not finished
_STARTING_SPACE_ = 4
_BLOCK_HEIGHT_ = 12

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def generate_excel_file(data, name, literals_synsets) -> None:

    count_total_sentences_to_fill = 0

    # Cream xlsx-ul
    workbook = xlsxwriter.Workbook( _OUTPUT_PATH_ + name )
    worksheet = workbook.add_worksheet()
    vcenter = workbook.add_format()
    vcenter.set_align('vcenter')
    worksheet.set_column('A:XFD', None, vcenter)

    # Widen first and second column and center text
    vcenter.set_align('vcenter')
    worksheet.set_column('A:XFD', None, vcenter)
    worksheet.set_column("A:A", 25 , vcenter)
    worksheet.set_column("B:B", 200, vcenter)
    worksheet.set_default_row(26)
    worksheet.set_row(0,50)

    # Colorarea
    sentence_already_there = []
    sentence_color_to_fill = []
    synset_format = workbook.add_format({ 'font_color':'#e3e3e3'})
    synset_format.set_align('center')
    synset_format.set_align('vcenter')
    name_color = workbook.add_format({'bold':True, 'font_color':'black', "bg_color":"#ffc2c2", "align":"vcenter"})
    name_to_fill_color = workbook.add_format({'bold':True, 'font_color':'black', "bg_color":"#ffd9d9", "align":"vcenter"})
    divider_color = workbook.add_format({'bold':True, 'bg_color':'#808080', "align":"vcenter"})
    definition_color = workbook.add_format({'bold':True, 'font_color':'black', "bg_color":"#FFB266", "align":"vcenter", 'text_wrap': True})
    sentence_color_to_fill.append(workbook.add_format({'bold':True, "bg_color":"A9E3FF", "align":"vcenter", 'text_wrap': True}))
    sentence_color_to_fill.append(workbook.add_format({'bold':True, "bg_color":"90DBFF", "align":"vcenter", 'text_wrap': True}))
    sentence_already_there.append(workbook.add_format({'bold':True, 'bg_color':'#AFFFAF', "align":"vcenter", 'text_wrap': True}))
    sentence_already_there.append(workbook.add_format({'bold':True, 'bg_color':'#9BFF9B', "align":"vcenter", 'text_wrap': True}))
    literal_color = workbook.add_format({'bold':True, 'bg_color':'#ff5252', "align":"vcenter", 'text_wrap': True})
    literal_color.set_align('center')

    # Lock every cell, unlock the blank sentences cells afterwards NOT WORKING
    if _LOCK_ == True:
        worksheet.protect()
        sentence_already_there[0].set_locked(False)
        sentence_already_there[1].set_locked(False)

    # Adaugăm primul divider
    worksheet.write(1, 0, "", divider_color)
    worksheet.write(1, 1, "", divider_color)

    # Rândul cu numele si prenumele
    worksheet.write(2, 0, "  Numele și Prenumele:", name_color)
    worksheet.write(2, 1, "  <introduceți numele aici>", name_to_fill_color)

    # Pentru fiecare pereche (literal, synset) umplem fisierul excel cu blocuri de forma:
    '''
    LITERAL  |   LITERALI:GLOSA                              <- Rândul _BLOCK_HEIGHT_*(al câtelea bloc) + _STARTING_SPACE_
    _________|__________________________________________
    SYNSET   |   PROPOZITII EXISTENTE
    _________|__________________________________________
             |   SPATIU DE UMPLUT CU PROPOZITII
    _________| _________________________________________
    '''
    for index in range(0,len(literals_synsets)):
        literal = literals_synsets[index][0]
        synset = literals_synsets[index][1]

        # Filling blank sentences with color and unlock them
        glosa_start_row = _BLOCK_HEIGHT_*index + _STARTING_SPACE_
        for i in range(glosa_start_row,glosa_start_row + 11):
            worksheet.write(i , 1, "", sentence_color_to_fill[i%2])

        # Legenda
        worksheet.write(0 , 1, _LEGEND_, name_color)

        # Divider
        worksheet.write(glosa_start_row-1, 0, "", divider_color)
        worksheet.write(glosa_start_row-1, 1, "", divider_color)

        # Randul de pe care incepem sa scriem propozitiile
        literal_start_row = 1 + _BLOCK_HEIGHT_*index + _STARTING_SPACE_
        sentence_count = len(data[literal]["synsets"][synset])
        blank_sentence_count = 10 - sentence_count
        # Pentru fiecare synset_id introducem literal, glosa, synset si propozitiile pe pozițiile corespunzătoare
        # Synset_id este necesar pentru citirea de date
        literals_glosa = str(wn(synset).literals) + "  " + str(wn(synset).definition)
        worksheet.write(glosa_start_row, 0, literal, literal_color)
        worksheet.write(glosa_start_row + 1, 0, synset, synset_format)
        worksheet.write(glosa_start_row + 2, 0, blank_sentence_count, synset_format)
        worksheet.write(glosa_start_row , 1, literals_glosa, definition_color)

        # data[literal]["synsets"][syn_id] --- Ar trebui sa aiba cel mult 9 propozitii
        '''
        if sentence_count > 9:
                print("Error: 10 or more sentences for {} -- {} | synset -- literal".format(synset, literal))
        '''

        # Introducem propozitiile deja existente:
        for sen_index in range(sentence_count):
            sentence = data[literal]["synsets"][synset][sen_index]
            sen_row = literal_start_row + sen_index
            worksheet.write(sen_row , 1, sentence, sentence_already_there[sen_row%2])

        count_total_sentences_to_fill += 10 - sentence_count

    # In colțul stănga sus al fisierului scriem câte propoziții sunt de umplut, pentru a ține evidența voluntarilor
    worksheet.write(0 , 0, count_total_sentences_to_fill, synset_format)
    workbook.close()




if __name__ == "__main__":

    # Loading data
    wn = rwn.RoWordNet()
    if not os.path.exists(_OUTPUT_PATH_):
        os.makedirs(_OUTPUT_PATH_)
    path_to_json = os.path.join(_INPUT_PATH_, "literal2synset_data.json")

    with open(path_to_json, encoding="utf8") as f:
        data = json.load(f)

    '''
    JSON data format:
    {
    "estimație":{
        "score":3.555555555555556,
        "synsets":{
            "ENG30-05736149-n":[
            ],
            "ENG30-00875246-n":[
            ],
            "ENG30-05803379-n":[
                "În mai 2008, The Boston Globe a stabilit o {estimație} de 670.000 de persoane pentru acest grup.",
                "În 1925 a pus bazele teoriei {estimației}, care face parte din domeniul statisticii matematice.",
                "Folosind metoda variabilei cefeide, în 2004 a fost obținută o {estimație} de 2,51 ± 0,13 milioane de ani lumină.",
                "Deoarece ieșirile sunt aleatoare, ele pot fi considerate doar ca {estimații} ale caracteristicilor adevărate ale sistemului."
            ]
        }
    }
    "armată": { ... }
    ...
    }
    '''

    # Sortare dictionar in functie de score
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1]['score'], reverse=True)}

    literal_cnt = _MAX_LITERALS_
    file_cnt = 1
    literals = []
    total_sen_cnt = 0
    literals_synsets_list = []

    for literal in sorted_data:

        literal_cnt -= 1
        if file_cnt > _MAXIMUM_FILE_COUNT_:
            break
        if literal_cnt == 0:
            break

        # Ne interesează literali cu cel puțin 2 synseturi
        if len(sorted_data[literal]["synsets"]) > 1:
            for synset in sorted_data[literal]["synsets"]:

                # Generăm fișier când găsim _MINIMUM_SENTENCES_PER_FILE_ de propoziții "lipsă"
                if total_sen_cnt >= _MINIMUM_SENTENCES_PER_FILE_:
                    generate_excel_file(sorted_data, name=f"wsd_{file_cnt}.xlsx", literals_synsets=literals_synsets_list)
                    file_cnt += 1
                    total_sen_cnt = 0
                    literals_synsets_list = []
                    # Terminal ProgressBar
                    if (_MAX_LITERALS_ - literal_cnt)/_MAX_LITERALS_ > (file_cnt/_MAXIMUM_FILE_COUNT_):
                        printProgressBar(_MAX_LITERALS_ - literal_cnt, _MAX_LITERALS_, prefix = 'Progress:', suffix = 'Complete', length = 50)
                    else:
                        printProgressBar(file_cnt, _MAXIMUM_FILE_COUNT_, prefix = 'Progress:', suffix = 'Complete', length = 50)

                count_sentences = len(sorted_data[literal]["synsets"][synset])

                #Introducem doar 2-upluri cu mai putin de 10 propozitii per synset:
                if count_sentences < 10:
                    literals_synsets_list.append((literal, synset))
                    total_sen_cnt += 10 - count_sentences


