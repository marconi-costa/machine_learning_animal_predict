# pelo longo
# pernas curtas
# faz auau
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

porco1 = [0, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 0, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 1, 1]
cachorro3 = [1, 0, 1]

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0]  # 0 cachorro # 1 porco

modelo = LinearSVC() #cerebro vazio
modelo.fit(treino_x, treino_y) #encaixar esses dados no modelo

animal_misterioso = [1, 1, 1] # tem pelo longo perna curta e faz auau

print(modelo.predict([animal_misterioso])) # print 0 porque é cachorro

misterio1 = [1,1,1] # tem pelo longo perna curta e faz auau
misterio2 = [1,1,0] # tem pelo longo perna curta e não faz auau
misterio3 = [0,1,1] # não tem pelo longo perna curta e faz auau

testex = [misterio1, misterio2, misterio3]
testey =[0,1,1] # eu sei que apenas o primeiro é um cachorro

previsoes = modelo.predict(testex)
print(previsoes) # cachorro, porco, cachorro
print(accuracy_score(previsoes,testey)) # accuracy tive uma taxa de acerto de 66%