import pygame
import pygame.midi
import time

#初期条件
pygame.init()
pygame.midi.init()

#デバイスの出力
count = pygame.midi.get_count()
print("get_default_input_id:%d" % pygame.midi.get_default_input_id())
print("get_default_output_id:%d" % pygame.midi.get_default_output_id())
print("No:(interf, name, input, output, opened)")
for i in range(count):
    print("%d:%s" % (i, pygame.midi.get_device_info(i)))

#ポート番号は人によって変わる、2じゃないかも
player = pygame.midi.Output(2)
player.set_instrument(48,1)

#5回音鳴らす
#第1が音の高さ、第2が音の大きさ
for i in range(5):
    player.note_on(70, 120)
    time.sleep(2)

del player
pygame.midi.quit()