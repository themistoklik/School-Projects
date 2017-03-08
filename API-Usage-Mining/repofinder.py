from github import Github
import sys
username = sys.argv[1]
password = sys.argv[2]

g = Github(username, password)

libs = ['"org.encog"', '"cc.mallet"', '"edu.stanford.nlp"', '"deeplearning4j-core"']


def getlinks(results):
    results = sorted(results,key=lambda x: x.repository.size)
    return ['https://github.com/' + element.repository.full_name + '.git' for element in results]

def writelinks(links,libname):
    with open(libname+'.txt','w') as f:
        for element in links:
            f.write(element+'\n')

for lib in libs:
    print "LIB-> "+lib
    results = []
    search = g.search_code(lib + ' pom.xml', order='desc', l="Maven+POM")
    for page in range(0, 4):
        results.extend(search.get_page(page))
    #remove dups
    results = set(results)
    links = getlinks(results)
    writelinks(links,lib)

