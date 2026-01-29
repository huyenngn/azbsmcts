<script setup lang="ts">
import GoStone from './GoStone.vue'

const props = defineProps<{
  board: number[]
}>()

const emit = defineEmits<{
  (e: 'move', index: number): void
}>()
</script>

<template>
  <div class="self-center grid grid-cols-9 aspect-9/8 w-90 h-80 ml-10 sm:ml-0">
    <div
      v-for="(cell, index) in props.board"
      :key="index"
      class="relative bg-orange-200 border border-orange-500 aspect-square nth-[9n]:border-0 nth-[9n]:bg-transparent nth-last-[-n+9]:bg-transparent nth-last-[-n+9]:border-0"
    >
      <GoStone
        :playerId="cell"
        @click="
          () => {
            if (cell < 0) emit('move', (8 - Math.floor(index / 9)) * 9 + (index % 9))
          }
        "
      />
    </div>
  </div>
</template>
