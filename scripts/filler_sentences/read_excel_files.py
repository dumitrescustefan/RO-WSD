import json, os

def generate_excel_file() -> None:
    # scrie fisierul in folder output


if __name__ == "__main":
    os.makedirs("output", exists_ok=True)

    # incarci json

    

    # sortare dict
    sorted_dict = {k: v for k, v in sorted(x.items(), key=lambda item: item[1]['score'])}

    literal_cnt = 5000
    file_cnt = 0
    literals = []
    for key, value in sorted_dict:
        literal_cnt -= 1
        if literal_cnt == 0:
            break
        # adaugi literali ( numarand propozitiile lipsa )
        # cand prop lipsa > 100, scrii fisierul cu un nume unic
        if scrie_fisier:
            generate_excel_file(name=f"wsd_{file_cnt}.xlsx", literals=)
