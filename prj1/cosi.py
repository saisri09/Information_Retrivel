import math
sqrt_sums_dc = 0
dcvector=[1,1]
qcvector=[1.5,3]
for x in dcvector:
    square = dcvector[x] * dcvector[x]
    sqrt_sums_dc += square
try:
    reciprocalsqrt_doc = (1 / math.sqrt(sqrt_sums_dc))
except:
    reciprocalsqrt_doc = 0
# caclulate d1/sqrt(d1^2) [sqrt(d1^2) is calculated above as reciprocalsqrt_doc]
for x in dcvector:
    dcvector[x] *= reciprocalsqrt_doc
# caculate cosine similarity between the Query words and docuemnt words
i = 0;
cosinevector=qcvector.copy()
while i < qcvector.__len__():
    qw = list(qcvector.keys())[i]
    if (qw in dcvector):
        cosinevector[qw] = dcvector[qw] * qcvector[qw]
    else:
        cosinevector[qw] = 0
    i+=1
print(cosinevector)