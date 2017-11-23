mport numpy
import pandas

df=pandas.read_excel('/home/wjd/python/machinelearn/123.xls')
dic={'house':'c','marry':'c','income':'n','loan':'c'}

def deal(df,column='income'):
    return math.sqrt(df.ix[:,['income']].pow(2).sum()-df.ix[:,['income']].size*df.ix[:,['income']].mean().pow(2))

if dic['loan'].value=='c'
        for i in dic:
                init=df.ix[:,[i,'loan']
                if dic[i]=='c':
                        a=init.groupby(['loan',i]).size().unstack().fillna(0)
                        columns=a.columns
                        result=1000
                        rvalue=str
                        for j in columns:
                                b=a.drop(j,axis=1).sum(axis=1)
                                c=a.ix[:,[j]]
                                ta=0
                                tb=0 
                                if dic['loan']=='c':
                                        for s in range(0,c.size):
                                                ta+=pow(b.ix[s]/b.sum(),2)
                                                tb+=pow(c.ix[s]/c.sum(),2)
                                        ta=1-ta
                                        tb=1-tb
                                        if result>ta*(b.sum()/(b.sum()+c.sum()))+tb*(c.sum()/(b.sum()+c.sum())):
                                                result=ta*(b.sum()/(b.sum()+c.sum()))+tb*(c.sum()/(b.sum()+c.sum()))
                                                rvalue=j
                                #else if dic['loan']=='n':
                                #       if result>init.groupby([i]).apply(deal).sum():
                                #               result=init.groupby([i]).apply(deal).sum()
                        
                else if dic[i]=='n':
                        columns=a.columns
                        result=1000
                        rvalue=int
                        for j in range(1,len(columns)):
                                l=columns[:j]
                                r=columns[j:]
                                b=a.ix[:,l].sum(axis=1)
                                c=a.ix[:,r].sum(axis=1)
                                ta=0
                                tb=0
                                for s in range(0,b.size):
                                        ta+=pow(b.ix[s]/b.sum(),2)
                                        tb+=pow(c.ix[s]/c.sum(),2)
                                ta=1-ta
                                tb=1-tb
                                if result>ta*(b.sum()/(b.sum()+c.sum()))+tb*(c.sum()/(b.sum()+c.sum())):
                                        result=ta*(b.sum()/(b.sum()+c.sum()))+tb*(c.sum()/(b.sum()+c.sum()))
                                        rvalue=columns[j-1]



else if dic['loan'].value=='n':
        for i in dic:
                init=df.ix[:,[i,'loan']]
                result=10000
                if dic[i]=='c':
                        rvalue=str
                        for j in set(tmp[i]):
                                ta=math.sqrt(init[init[i]==j].ix[:,[j]].pow(2).sum()-init[init[i]==j].ix[:,[j]].size*init[init[i]==j].ix[:,[j]].mean())
                                tb=math.sqrt(init[init[i]!=j].ix[:,[j]].pow(2).sum()-init[init[i]!=j].ix[:,[j]].size*init[init[i]!=j].ix[:,[j]].mean())
                                if result>(ta+tb):
                                        result=ta+tb
                                        rvalue=j
                else if dic[i]=='n':
                        rvalue=int
                        a=init.sort_values(i)[i].values
                        for j in a:
                                ta=math.sqrt(init[init[i]>j].ix[:,[j]].pow(2).sum()-init[init[i]>j].ix[:,[j]].size*init[init[i]>j].ix[:,[j]].mean())

     				tb=math.sqrt(init[init[i]<=j].ix[:,[j]].pow(2).sum()-init[init[i]<=j].ix[:,[j]].size*init[init[i]<=j].ix[:,[j]].mean())
                                if result>(ta+tb):
                                       result=ta+tb
                                       rvalue=j

































































