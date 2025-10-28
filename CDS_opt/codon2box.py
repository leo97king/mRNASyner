def cds2box(cds_file,box_file):
    ##Convert the CDS sequence to an amino acid sequence, and use BIO annotation to label the codon box.
    f1 = open(cds_file,'r') #input
    f2 = open(box_file,'w') #output
    

    line = ''
    i = 0
    count = 0
    err=0
    for line in f1:
        line = line.strip('\n').replace('U','T')
        while i <len(line):
            s=line[i]+line[i+1]+line[i+2]
            if s == 'TTT':
                o = 'F'
                w='I-a'
            elif s == 'TTC':
                o = 'F'
                w = 'I-b'
            elif s == 'TTA':
                o = 'L'
                w = 'I-c'
            elif s == 'TTG':
                o = 'L'
                w = 'I-d'
            elif s == 'TCT':
                o = 'S'
                w = 'I-b'
            elif s == 'TCC':
                o = 'S'
                w = 'I-f'
            elif s == 'TCA':
                o = 'S'
                w = 'I-h'
            elif s == 'TCG':
                o = 'S'
                w = 'I-g'
            elif s == 'TAT':
                o = 'Y'
                w = 'I-c'
            elif s == 'TAC':
                o = 'Y'
                w = 'I-h'
            elif s == 'TAA':
                o = 'X'
                w = 'O-w'
            elif s == 'TAG':
                o = 'X'
                w = 'O-w'
            elif s == 'TGT':
                o = 'C'
                w = 'I-d'
            elif s == 'TGC':
                o = 'C'
                w = 'I-g'
            elif s == 'TGA':
                o = 'X'
                w = 'O-w'
            elif s == 'TGG':
                o = 'W'
                w = 'I-k'
            elif s == 'CTT':
                o = 'L'
                w = 'I-b'
            elif s == 'CTC':
                o = 'L'
                w = 'I-f'
            elif s == 'CTA':
                o = 'L'
                w = 'I-h'
            elif s == 'CTG':
                o = 'L'
                w = 'I-g'
            elif s == 'CCT':
                o = 'P'
                w = 'I-f'
            elif s == 'CCC':
                o = 'P'
                w = 'I-l'
            elif s == 'CCA':
                o = 'P'
                w = 'I-m'
            elif s == 'CCG':
                o = 'P'
                w = 'I-n'
            elif s == 'CAT':
                o = 'H'
                w = 'I-h'
            elif s == 'CAC':
                o = 'H'
                w = 'I-m'
            elif s == 'CAA':
                o = 'Q'
                w = 'I-o'
            elif s == 'CAG':
                o = 'Q'
                w = 'I-r'
            elif s == 'CGT':
                o = 'R'
                w = 'I-g'
            elif s == 'CGC':
                o = 'R'
                w = 'I-n'
            elif s == 'CGA':
                o = 'R'
                w = 'I-r'
            elif s == 'CGG':
                o = 'R'
                w = 'I-s'
            elif s == 'ATT':
                o = 'I'
                w = 'I-c'
            elif s == 'ATC':
                o = 'I'
                w = 'I-h'
            elif s == 'ATA':
                o = 'I'
                w = 'I-i'
            elif s == 'ATG' and i == 0:
                o = 'M'
                w = 'B-j'
            elif s == 'ATG' and i != 0:
                o = 'M'
                w = 'I-j'
            elif s == 'ACT':
                o = 'T'
                w = 'I-h'
            elif s == 'ACC':
                o = 'T'
                w = 'I-m'
            elif s == 'ACA':
                o = 'T'
                w = 'I-o'
            elif s == 'ACG':
                o = 'T'
                w = 'I-r'
            elif s == 'AAT':
                o = 'N'
                w = 'I-i'
            elif s == 'AAC':
                o = 'N'
                w = 'I-o'
            elif s == 'AAG':
                o = 'K'
                w = 'I-u'
            elif s == 'AAA':
                o = 'K'
                w = 'I-t'
            elif s == 'AGT':
                o = 'S'
                w = 'I-j'
            elif s == 'AGC':
                o = 'S'
                w = 'I-r'
            elif s == 'AGA':
                o = 'R'
                w = 'I-u'
            elif s == 'AGG':
                o = 'R'
                w = 'I-q'
            elif s == 'GTT':
                o = 'V'
                w = 'I-d'
            elif s == 'GTC':
                o = 'V'
                w = 'I-g'
            elif s == 'GTA':
                o = 'V'
                w = 'I-j'
            elif s == 'GTG':
                o = 'V'
                w = 'I-k'
            elif s == 'GCT':
                o = 'A'
                w = 'I-g'
            elif s == 'GCC':
                o = 'A'
                w = 'I-n'
            elif s == 'GCA':
                o = 'A'
                w = 'I-r'
            elif s == 'GCG':
                o = 'A'
                w = 'I-s'
            elif s == 'GAT':
                o = 'D'
                w = 'I-j'
            elif s == 'GAC':
                o = 'D'
                w = 'I-r'
            elif s == 'GAA':
                o = 'E'
                w = 'I-u'
            elif s == 'GAG':
                o = 'E'
                w = 'I-q'
            elif s == 'GGT':
                o = 'G'
                w = 'I-k'
            elif s == 'GGC':
                o = 'G'
                w = 'I-s'
            elif s == 'GGA':
                o = 'G'
                w = 'I-q'
            elif s == 'GGG':
                o = 'G'
                w = 'I-p'
            else:
                err = err+1
            s2=o+" "+w
            f2.writelines(s2+"\n")
            i=i+3
            if o=='X':
                f2.write('\n')
                anino=''
        s1 = ''
        s2=''
        line = ''
        flag = 0
        i=0
    f2.close()
    f1.close()

def box2cds(box_file,cds_file):
    #Convert the codon box to cds sequence
    f = open(box_file,'r') 
    f2= open(cds_file,'w') 

    line = ''
    count = 1
    count1=0
    s = ''
    c=0
    err = []
    col=[]
    sum=0
    for line in f:
        line=line.strip('\n')

        if(line == ''):
            f2.writelines(s + '\n')
            s = ''
            count = count + 1
            continue
        if(line[0] == '#'):
            continue

        else:
            count1 = count1+1
            col=line.split('\t')
            if sum<=2:
                sum+=1
            if(col[0] == 'F' and col[2]== 'I-a'):
                s = s + 'UUU'
            elif(col[0] == 'F' and col[2]== 'I-b'):
                s = s + 'UUC'
            elif(col[0] == 'L' and col[2]== 'I-c'):
                s = s + 'UUA'
            elif(col[0] == 'L' and col[2]== 'I-d'):
                s = s + 'UUG'
            elif(col[0] == 'L' and col[2]== 'I-b'):
                s = s + 'CUU'
            elif(col[0] == 'L' and col[2]== 'I-f'):
                s = s + 'CUC'
            elif(col[0] == 'L' and col[2]== 'I-h'):
                s = s + 'CUA'
            elif(col[0] == 'L' and col[2]== 'I-g'):
                s = s + 'CUG'
            elif(col[0] == 'I' and col[2]== 'I-c'):
                s = s + 'AUU'
            elif(col[0] == 'I' and col[2]== 'I-h'):
                s = s + 'AUC'
            elif(col[0] == 'I' and col[2]== 'I-i'):
                s = s + 'AUA'
            elif(col[0] == 'V' and col[2]== 'I-d'):
                s = s + 'GUU' 
            elif(col[0] == 'V' and col[2]== 'I-g'):
                s = s + 'GUC'
            elif(col[0] == 'V' and col[2]== 'I-j'):
                s = s + 'GUA'     
            elif(col[0] == 'V' and col[2]== 'I-k'):
                s = s + 'GUG'
            elif(col[0] == 'S' and col[2]== 'I-b'):
                s = s + 'UCU'
            elif(col[0] == 'S' and col[2]== 'I-f'):
                s = s + 'UCC'
            elif(col[0] == 'S' and col[2]== 'I-h'):
                s = s + 'UCA'
            elif(col[0] == 'S' and col[2]== 'I-g'):
                s = s + 'UCG'
            elif(col[0] == 'S' and col[2]== 'I-j'):
                s = s + 'AGU'
            elif(col[0] == 'S' and col[2]== 'I-r'):
                s = s + 'AGC'
            elif(col[0] == 'P' and col[2]== 'I-f'):
                s = s + 'CCU'
            elif(col[0] == 'P' and col[2]== 'I-l'):
                s = s + 'CCC'
            elif(col[0] == 'P' and col[2]== 'I-m'):
                s = s + 'CCA'
            elif(col[0] == 'P' and col[2]== 'I-n'):
                s = s + 'CCG'
            elif(col[0] == 'T' and col[2]== 'I-h'):
                s = s + 'ACU'
            elif(col[0] == 'T' and col[2]== 'I-m'):
                s = s + 'ACC'
            elif(col[0] == 'T' and col[2]== 'I-o'):
                s = s + 'ACA'
            elif(col[0] == 'T' and col[2]== 'I-r'):
                s = s + 'ACG'
            elif(col[0] == 'A' and col[2]== 'I-g'):
                s = s + 'GCU'
            elif(col[0] == 'A' and col[2]== 'I-n'):
                s = s + 'GCC'
            elif(col[0] == 'A' and col[2]== 'I-r'):
                s = s + 'GCA'
            elif(col[0] == 'A' and col[2]== 'I-s'):
                s = s + 'GCG'
            elif(col[0] == 'Y' and col[2]== 'I-c'):
                s = s + 'UAU'
            elif(col[0] == 'Y' and col[2]== 'I-h'):
                s = s + 'UAC'
            elif(col[0] == 'X' and col[2]== 'O-w'):
                s = s + 'UGA'
            elif(col[0] == 'H' and col[2]== 'I-h'):
                s = s + 'CAU'
            elif(col[0] == 'H' and col[2]== 'I-m'):
                s = s + 'CAC'
            elif(col[0] == 'Q' and col[2]== 'I-o'):
                s = s + 'CAA'
            elif(col[0] == 'Q' and col[2]== 'I-r'):
                s = s + 'CAG'
            elif(col[0] == 'N' and col[2]== 'I-i'):
                s = s + 'AAU'
            elif(col[0] == 'N' and col[2]== 'I-o'):
                s = s + 'AAC'
            elif(col[0] == 'K' and col[2]== 'I-t'):
                s = s + 'AAA'
            elif(col[0] == 'K' and col[2]== 'I-u'):
                s = s + 'AAG'
            elif(col[0] == 'D' and col[2]== 'I-j'):
                s = s + 'GAU'
            elif(col[0] == 'D' and col[2]== 'I-r'):
                s = s + 'GAC'
            elif(col[0] == 'E' and col[2]== 'I-u'):
                s = s + 'GAA'
            elif(col[0] == 'E' and col[2]== 'I-q'):
                s = s + 'GAG'
            elif(col[0] == 'C' and col[2]== 'I-d'):
                s = s + 'UGU'
            elif(col[0] == 'C' and col[2]== 'I-g'):
                s = s + 'UGC'
            elif(col[0] == 'R' and col[2]== 'I-g'):
                s = s + 'CGU'
            elif(col[0] == 'R' and col[2]== 'I-n'):
                s = s + 'CGC'
            elif(col[0] == 'R' and col[2]== 'I-r'):
                s = s + 'CGA'
            elif(col[0] == 'R' and col[2]== 'I-s'):
                s = s + 'CGG'
            elif(col[0] == 'R' and col[2]== 'I-u'):
                s = s + 'AGA'
            elif(col[0] == 'R' and col[2]== 'I-q'):
                s = s + 'AGG'
            elif(col[0] == 'G' and col[2]== 'I-k'):
                s = s + 'GGU'
            elif(col[0] == 'G' and col[2]== 'I-s'):
                s = s + 'GGC'
            elif(col[0] == 'G' and col[2]== 'I-q'):
                s = s + 'GGA'
            elif(col[0] == 'G' and col[2]== 'I-p'):
                s = s + 'GGG'
            elif(col[0] == 'M' and col[2]== 'B-j'):
                s = s + 'AUG'
            elif(col[0] == 'M' and col[2]== 'I-j'):
                s = s + 'AUG'
            elif(col[0] == 'W' and col[2]== 'I-k'):
                s = s + 'UGG'

            
    f.close()
    f2.close()