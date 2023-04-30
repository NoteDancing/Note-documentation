from tensorflow import function
from tensorflow import data as tf_data
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import os
import time


'''
example:
import nspn.kernel_reduced as k   #import kernel
import tensorflow as tf
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn)   #start kernel
kernel.platform=tf
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.train(500)
'''
class kernel:
    def __init__(self,nn=None,save_episode=False):
        self.nn=nn
        self.platform=None
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        self.episode_set=[]
        self.epsilon=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.trial_count=None
        self.criterion=None
        self.reward_list=[]
        self.max_episode_count=None
        self.save_episode=save_episode
        self.filename='save.dat'
        self.loss=None
        self.loss_list=[]
        self.sc=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def action_vec(self):
        if self.epsilon!=None:
            self.action_one=np.ones(self.action_count,dtype=np.int8)
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None,end_loss=None):
        if epsilon!=None:
            self.epsilon=epsilon
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
        if update_step!=None:
            self.update_step=update_step
        if trial_count!=None:
            self.trial_count=trial_count
        if criterion!=None:
            self.criterion=criterion
        if end_loss!=None:
            self.end_loss=end_loss
        self.action_vec()
        return
    
    
    def epsilon_greedy_policy(self,s):
        action_prob=self.action_one*self.epsilon/len(self.action_one)
        try:
            if self.platform.DType!=None:
                best_a=np.argmax(self.nn.nn.fp(s))
                action_prob[best_a]+=1-self.epsilon
        except AttributeError:
            best_a=self.nn.nn(s).argmax()
            action_prob[best_a.numpy()]+=1-self.epsilon
        return action_prob
    
    
    @function(jit_compile=True)
    def tf_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        with self.platform.GradientTape(persistent=True) as tape:
            loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        try:
            gradient=self.nn.gradient(tape,loss)
            try:
                self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
            except AttributeError:
                self.nn.opt(gradient)
        except AttributeError:
            try:
                if self.nn.nn!=None:
                    gradient=tape.gradient(loss,self.nn.param)
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
            except AttributeError:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
                    self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                    self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
        return loss
    
    
    def pytorch_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        self.nn.backward(loss)
        self.nn.opt()
        return loss
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        try:
            if self.platform.DType!=None: 
                loss=self.tf_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        except AttributeError:
            loss=self.pytorch_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        return loss
    
    
    def opt_ol(self,state,action,next_state,reward,done):
        try:
            if self.platform.DType!=None: 
                loss=self.tf_opt(state,action,next_state,reward,done)
        except AttributeError:
            loss=self.pytorch_opt(state,action,next_state,reward,done)
        return loss
    
    
    def pool(self,s,a,next_s,r,done):
        if type(self.state_pool)!=np.ndarray and self.state_pool==None:
            self.state_pool=s
            if type(a)==int:
                a=np.array(a)
                self.action_pool=np.expand_dims(a,axis=0)
            else:
                self.action_pool=a
            self.next_state_pool=np.expand_dims(next_s,axis=0)
            self.reward_pool=np.expand_dims(r,axis=0)
            self.done_pool=np.expand_dims(done,axis=0)
        else:
            self.state_pool=np.concatenate((self.state_pool,s),0)
            if type(a)==int:
                a=np.array(a)
                self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0)
            else:
                self.action_pool=np.concatenate((self.action_pool,a),0)
            self.next_state_pool=np.concatenate((self.next_state_pool,np.expand_dims(next_s,axis=0)),0)
            self.reward_pool=np.concatenate((self.reward_pool,np.expand_dims(r,axis=0)),0)
            self.done_pool=np.concatenate((self.done_pool,np.expand_dims(done,axis=0)),0)
        if len(self.state_pool)>self.pool_size:
            self.state_pool=self.state_pool[1:]
            self.action_pool=self.action_pool[1:]
            self.next_state_pool=self.next_state_pool[1:]
            self.reward_pool=self.reward_pool[1:]
            self.done_pool=self.done_pool[1:]
        return
    
    
    def _train(self):
        if len(self.state_pool)<self.batch:
            return np.array(0.)
        else:
            loss=0
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            j=0
            train_ds=tf_data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
            try:
                self.nn.bc=0
            except AttributeError:
                pass
            for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                try:
                    if self.platform.DType!=None:
                        pass
                except AttributeError:
                    state_batch=state_batch.numpy()
                    action_batch=action_batch.numpy()
                    next_state_batch=next_state_batch.numpy()
                    reward_batch=reward_batch.numpy()
                    done_batch=done_batch.numpy()
                batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                loss+=batch_loss
                j+=1
                try:
                    self.nn.bc+=1
                except AttributeError:
                    pass
            if self.update_step!=None:
                if self.sc%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
        try:
            if self.platform.DType!=None:
                loss=loss.numpy()/batches
        except AttributeError:
            loss=loss.detach().numpy()/batches
        return loss
    
    
    def train_(self):
        episode=[]
        self.reward=0
        s=self.nn.env(initial=True)
        if self.episode_step==None:
            while True:
                try:
                    if self.nn.nn!=None:
                        try:
                            if self.platform.DType!=None:
                                s=np.expand_dims(s,axis=0)
                                action_prob=self.epsilon_greedy_policy(s)
                                a=np.random.choice(self.action_count,p=action_prob)
                        except AttributeError:
                            s=np.expand_dims(s,axis=0)
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                            action_prob=self.epsilon_greedy_policy(s)
                            a=np.random.choice(self.action_count,p=action_prob)
                        next_s,r,done=self.nn.env(a)
                        r=np.array(r,dtype=np.float32)
                        done=np.array(done,dtype=np.float32)
                except AttributeError:
                    try:
                        if self.nn.action!=None:
                            try:
                                if self.platform.DType!=None:
                                    s=np.expand_dims(s,axis=0)
                                    a=self.nn.action(s).numpy()
                            except AttributeError:
                                    s=np.expand_dims(s,axis=0)
                                    s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                                    a=self.nn.action(s).detach().numpy()
                    except AttributeError:
                        try:
                            if self.platform.DType!=None:
                                s=np.expand_dims(s,axis=0)
                                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
                        except AttributeError:
                            s=np.expand_dims(s,axis=0)
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                            a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()
                        next_s,r,done=self.nn.env(a)
                        r=np.array(r,dtype=np.float32)
                        done=np.array(done,dtype=np.float32)
                self.pool(s,a,next_s,r,done)
                self.reward=r+self.reward
                loss=self._train()
                self.sc+=1
                if done:
                    if self.save_episode==True:
                        episode=[s,a,next_s,r]
                    self.reward_list.append(self.reward)
                    return loss,episode,done
                elif self.save_episode==True:
                    episode=[s,a,next_s,r]
                s=next_s
        else:
            for _ in range(self.episode_step):
                try:
                    if self.nn.nn!=None:
                        try:
                            if self.platform.DType!=None:
                                s=np.expand_dims(s,axis=0)
                                action_prob=self.epsilon_greedy_policy(s)
                                a=np.random.choice(self.action_count,p=action_prob)
                        except AttributeError:
                            s=np.expand_dims(s,axis=0)
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                            action_prob=self.epsilon_greedy_policy(s)
                            a=np.random.choice(self.action_count,p=action_prob)
                        next_s,r,done=self.nn.env(a)
                        r=np.array(r,dtype=np.float32)
                        done=np.array(done,dtype=np.float32)
                except AttributeError:
                    try:
                        if self.nn.action!=None:
                            try:
                                if self.platform.DType!=None:
                                    s=np.expand_dims(s,axis=0)
                                    a=self.nn.action(s).numpy()
                            except AttributeError:
                                s=np.expand_dims(s,axis=0)
                                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                                a=self.nn.action(s).detach().numpy()
                    except AttributeError:
                        try:
                            if self.platform.DType!=None:
                                s=np.expand_dims(s,axis=0)
                                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
                        except AttributeError:
                            s=np.expand_dims(s,axis=0)
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                            a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()
                    next_s,r,done=self.nn.env(a)
                    r=np.array(r,dtype=np.float32)
                    done=np.array(done,dtype=np.float32)
                self.pool(s,a,next_s,r,done)
                self.reward=r+self.reward
                loss=self._train()
                self.sc+=1
                if done:
                    if self.save_episode==True:
                        episode=[s,a,next_s,r]
                    self.reward_list.append(self.reward)
                    return loss,episode,done
                elif self.save_episode==True:
                    episode=[s,a,next_s,r]
                s=next_s
        self.reward_list.append(self.reward)
        return loss,episode,done
    
    
    def train(self,episode_count,save=None,one=True,p=None,s=None):
        avg_reward=None
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if s==None:
            self.s=1
            self.file_list=None
        else:
            self.s=s-1
            self.file_list=[]
        if episode_count!=None:
            for i in range(episode_count):
                t1=time.time()
                loss,episode,done=self.train_()
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            self._time=self.total_time-int(self.total_time)
                            if self._time<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.6f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                self.loss=loss
                self.loss_list.append(loss)
                self.total_episode+=1
                if episode_count%10!=0:
                    p=episode_count-episode_count%self.p
                    p=int(p/self.p)
                    s=episode_count-episode_count%self.s
                    s=int(s/self.s)
                else:
                    p=episode_count/(self.p+1)
                    p=int(p)
                    s=episode_count/(self.s+1)
                    s=int(s)
                if p==0:
                    p=1
                if s==0:
                    s=1
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                if save!=None and i%s==0:
                    self.save(self.total_episode,one)
                if self.save_episode==True:
                    if done:
                        episode.append('done')
                    self.episode_set.append(episode)
                    if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count:
                        self.save_episode=False
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                loss,episode,done=self.train_()
                if self.trial_count!=None:
                    if len(self.reward_list)==self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            self._time=self.total_time-int(self.total_time)
                            if self._time<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.6f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.total_episode+=1
                if episode_count%10!=0:
                    p=episode_count-episode_count%self.p
                    p=int(p/self.p)
                    s=episode_count-episode_count%self.s
                    s=int(s/self.s)
                else:
                    p=episode_count/(self.p+1)
                    p=int(p)
                    s=episode_count/(self.s+1)
                    s=int(s)
                if p==0:
                    p=1
                if s==0:
                    s=1
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                if save!=None and i%s==0:
                    self.save(self.total_episode,one)
                if self.save_episode==True:
                    if done:
                        episode.append('done')
                    self.episode_set.append(episode)
                    if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count:
                        self.save_episode=False
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                t2=time.time()
                self.time+=(t2-t1)
        self._time=self.time-int(self.time)
        if self._time<0.5:
            self.total_time=int(self.time)
        else:
            self.total_time=int(self.time)+1
        self.total_time+=self.time
        print('last loss:{0:.6f}'.format(loss))
        print('last reward:{0}'.format(self.reward))
        print()
        print('time:{0}s'.format(self.time))
        return
    
    
    def train_ol(self):
        while True:
            if self.stop_flag==True:
                return
            if self.save_flag==True:
                self.save()
            self.suspend_func()
            data=self.nn.ol()
            if data=='stop':
                return
            elif data=='suspend':
                while True:
                    self.nn.suspend=True
                    while True:
                        if self.nn.suspend==False:
                            break
                    continue
            loss=self.opt_ol(data[0],data[1],data[2],data[3],data[4])
            loss=loss.numpy()
            self.nn.train_loss_list.append(loss)
            if len(self.nn.train_acc_list)==self.nn.max_length:
                del self.nn.train_acc_list[0]
            try:
                self.nn.c+=1
            except AttributeError:
                pass
        return
    
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_episode),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_episode),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def save_e(self):
        episode_file=open('episode.dat','wb')
        pickle.dump(self.episode_set,episode_file)
        episode_file.close()
        return
    
    
    def save(self,i=None,one=True):
        if one==True:
            output_file=open(self.filename,'wb')
            if self.save_episode==True:
                episode_file=open('episode.dat','wb')
                pickle.dump(self.episode_set,episode_file)
                episode_file.close()
        else:
            filename=self.filename.replace(self.filename[self.filename.find('.'):],'-{0}.dat'.format(i))
            output_file=open(filename,'wb')
            self.file_list.append([filename])
            if self.save_episode==True:
                episode_file=open('episode-{0}.dat'.format(i),'wb')
                pickle.dump(self.episode_set,episode_file)
                episode_file.close()
            if self.save_episode==True:
                self.file_list.append([filename,'episode-{0}.dat'])
                if len(self.file_list)>self.s+1:
                    os.remove(self.file_list[0][0])
                    os.remove(self.file_list[0][1])
                    del self.file_list[0]
            else:
                self.file_list.append([filename])
                if len(self.file_list)>self.s+1:
                    os.remove(self.file_list[0][0])
                    del self.file_list[0]
        try:
            if self.platform.DType!=None:
                try:
                    pickle.dump(self.nn,output_file)
                except:
                    opt=self.nn.opt
                    self.nn.opt=None
                    pickle.dump(self.nn,output_file)
                    self.nn.opt=opt
        except AttributeError:
            pass
        try:
            pickle.dump(self.platform.keras.optimizers.serialize(opt),output_file)
        except:
            pickle.dump(self.nn.serialize(),output_file)
        else:
            pickle.dump(None,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.max_episode_count,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.reward_list,output_file)
        pickle.dump(self.loss,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.sc,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,e_path=None):
        input_file=open(s_path,'rb')
        if e_path!=None:
            episode_file=open(e_path,'rb')
            self.episode_set=pickle.load(episode_file)
            episode_file.close()
        self.nn=pickle.load(input_file)
        opt_serialized=pickle.load(input_file)
        try:
            self.nn.opt=self.platform.keras.optimizers.deserialize(opt_serialized)
        except:
            self.nn.deserialize(opt_serialized)
        else:
            pass
        self.epsilon=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.max_episode_count=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.reward_list=pickle.load(input_file)
        self.loss=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.sc=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return