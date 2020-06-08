import cran
import query
from cranqry import loadCranQry
from index import InvertedIndex, test
from query import QueryProcessor

print("***************Test Cases Running for Index File****************")
invertedobj=InvertedIndex()
test(invertedobj)

print("***************Test Cases Running for Query File****************")
# load documents
inputdocument = cran.CranFile("cran.all")
# load the index file saved at from part 1
index = InvertedIndex().load("index_file")
# load query processed files
queries = loadCranQry("query.text")

qp = QueryProcessor(queries, index, inputdocument, 29)
query.test(qp)

qp= QueryProcessor(queries,index,inputdocument,29)
qp.vectorQuery(3)