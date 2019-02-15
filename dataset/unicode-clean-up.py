with open('raw_data_by_year/train/twitter-2015train-A.txt', 'r') as f:
    with open('raw_data_by_year/train/twitter-2015train-A-fixed.txt', 'w', encoding="utf-8") as write_f:
        for line in f:
            fixed_line = bytes(line, 'utf-8').decode('unicode-escape')
            write_f.write(fixed_line)
