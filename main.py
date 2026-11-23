from nn.Value import Value 
from nn.nn import MLP
from draw_dot.draw_dot import draw_dot


n = MLP(3,[4,4,1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys=[1.0,-1.0,-1.0,1.0]


for k in range(1000):
    # forward
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # tuning   
    
    for p in n.parameters():
        p.data += -0.01*p.grad

    print(k, loss.data)



# print(ypred)
draw_dot(loss)

from visualization.frontend import launch

launch(
    model_factory=lambda: MLP(3, [4, 4, 1]),
    xs=xs,
    ys=ys
)