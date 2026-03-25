from src.replay_buffer.replay_buffer import ReplayBuffer

print('Testing replay buffer...')

buf = ReplayBuffer(max_size=1000, algorithm='qlearning', task='default')

# Add 100 fake transitions
for i in range(100):
    buf.add(i, 0, -0.1, i+1, False)

print(f'✅ Buffer size: {len(buf)}')

# Sample a batch
states, actions, rewards, next_states, dones = buf.sample(32)
print(f'✅ Sampled batch of 32')

# Save to disk
buf.save()
print('✅ Saved to disk')

print('\n🎉 Replay buffer working!')