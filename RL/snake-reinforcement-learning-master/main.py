from double_q_learning_trainer import double_q_learning_T
import cv2
from matplot import pplot



if __name__ == "__main__":

  
    num_mode= int(input("1. Training을 진행한다.\n2. Play을 진행한다.\n3. Reward,Loss에 대한 그래프를 그린다.\n4. Log파일 삭제\n" ))
    
    if num_mode==1:
     
        num_episodes =int(input("학습을 진행할 episode의 수량을 입력해주세요 : "))

        trainer = double_q_learning_T(
            level_filepath='levels/9x9_empty.yml',
            episodes=num_episodes,
            initial_epsilon=1.,
            min_epsilon=0.1,
            exploration_ratio=0.5,
            max_steps=2000,
            render_freq=500,
            enable_render=True,
            render_fps=20,
            save_dir='checkpoints',
            enable_save=True,
            save_freq=500,
            gamma=0.99,
            batch_size=64,
            min_replay_memory_size=1000,
            replay_memory_size=100000,
            target_update_freq=5,
            seed=42
        )

        checkpoint = None
        if checkpoint is not None:
            trainer.load(checkpoint)

        trainer.train()


    if num_mode==2:
        trainer = double_q_learning_T(
            level_filepath='levels/9x9_empty.yml',
            max_steps=2000,
            save_dir='checkpoints',
            seed=42
        )
        trainer.load('best')

        trainer.preview(
            render_fps=10,
            disable_exploration=True,
        )

    if num_mode ==3:
        #여기가 그래프 호출
        temp = pplot()
        temp.reward_loss_plot()
        a = cv2.imread('save_averageloss.png')
        cv2.imshow('title', a)
        cv2.waitKey()

    if num_mode ==4:
        #그래프로 부터 읽어올 txt파일 내용 지움
        with open("test_plot.txt",'r+') as f:
            f.truncate(0)
        with open("test_main.txt",'r+') as f:
            f.truncate(0)