#1001
n = int(input())
count = 0
while(n!=1):
    if (n%2==0):
        n=n/2
    else:
        n=(3*n+1)/2
    count = count+1
print (count)

# 1002
Numstr = input()#Numstr用来保存输入的数字串,默认输入是字符串，所以不需要用str()强制类型转换
length = len(Numstr)
sum = 0
for i in range(length):
    sum = sum + eval(Numstr[i])#sum用来计算输入数字串各位数字之和,eval()功能是将字符串str当成有效的表达式来求值并返回计算结果
Sumstr = str(sum)#Sumstr用来将得到的数字之和字符串化，方便后续利用字典得到对应拼音
pinyin = {"0":"ling","1":"yi","2":"er","3":"san","4":"si","5":"wu","6":"liu","7":"qi","8":"ba","9":"jiu"}
for j in range(len(Sumstr)-1):
    print(pinyin.get(Sumstr[j]),end = " ")#输出第一位到倒数第二位，每一位后面带空格
print(pinyin.get(Sumstr[-1]))#最后一位单独输出，确保输出之后没有空格

# 1003
import re
n=input()
for i in range(int(n)):
    s=input()
    if re.match(r'A*PA+TA*',s): #在字符串中进行匹配
        a=re.split(r'[P|T]',s)  #以字符P,T进行分段
        if a[0]*len(a[1])==a[2]:  #条件判断
            print('YES')
        else:
            print('NO')
    else:
        print('NO')

# 1004
n=int(input())
name_list=[]
num_list=[]
score_list=[]
for i in range(n):
    item=list(input().split(' '))
    name_list.append(item[0])
    num_list.append(item[1])
    score_list.append(int(item[2]))
item_max=max(score_list)
max_index=score_list.index(item_max)
item_min=min(score_list)
min_index=score_list.index(item_min)
print(name_list[max_index]+" "+num_list[max_index])
print(name_list[min_index]+" "+num_list[min_index])

# 1004
n=input()
d={}
b=[]
for i in range(int(n)):
    a = input().split()
    d[int(a[2])]=a[0]+' '+a[1]
for key in d:
    b.append(key)
print(d[max(b)])
print(d[min(b)])

# 1005
k = int(input())
n = input().split()
numlist_1 = [int(n[i]) for i in range(k)]
numlist_2 = [int(n[i]) for i in range(k)]
covered = []
not_covered = []
for i in range(len(numlist_1)):
    while numlist_1[i] != 1:
        if numlist_1[i] % 2 == 0:
            numlist_1[i] = numlist_1[i] / 2
            covered.append(numlist_1[i])
        else:
            numlist_1[i] = (numlist_1[i] * 3 + 1) / 2
            covered.append(numlist_1[i])
for x in range(len(numlist_2)):
    if numlist_2[x] not in covered:
        not_covered.append(numlist_2[x])
not_covered.sort(reverse=True)  # 倒序
a = []
for tmp in not_covered:
    a.append(str(tmp))
print(' '.join(a))

# 1006
numstr = input()
hundreds = 0
tens = 0
if len(numstr)==3:
    hundreds = int(numstr[0])
    tens = int(numstr[1])
    digits = int(numstr[2])
elif len(numstr)==2:
    tens = int(numstr[0])
    digits = int(numstr[1])
else:
    digits = int(numstr[0])
fin=''
for i in range(digits):
    fin=fin+str(i+1)
print("B"*hundreds+"S"*tens+str(fin))










