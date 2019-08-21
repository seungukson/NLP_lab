class Dog:
    def __init__(self):
        self.age = 4

    def howManyLegs(self):
        return self.age
    def canItSing(self):
        return "no"

class DdangChil(Dog):
    def __init__(self,age,name,live):
        self.age = 9
        self.live = live
        self.name = name;
        super(DdangChil, self).__init__()


    def howOld(self):
        return self.age

    def whatName(self):
        return "DdangChil"

    def canItSing(self):
        return "Yes"

x = DdangChil()
print(x.age)
print(x.howOld())

y = DdangChil(10,"dangchil",'yeoksam')
z = DdangChil(3,"dangchil","kangNam")

y.canItSing()
y.age
z.age
z.live

abc = DdangChil(10,"dd","ab")
abc.howManyLegs()