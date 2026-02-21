<script setup lang="ts">
import GoStone from "./GoStone.vue"

const props = defineProps<{
  boardSize: number
  board: number[]
}>()

const emit = defineEmits<{
  (e: "move", index: number): void
}>()
</script>

<template>
  <div
    class="self-center grid ml-10 sm:ml-0"
    :style="{
      gridTemplateColumns: `repeat(${props.boardSize}, minmax(0, 1fr))`,
      aspectRatio: `${props.boardSize} / ${props.boardSize - 1}`,
      width: `${props.boardSize * 2.5}rem`,
      height: `${(props.boardSize - 1) * 2.5}rem`,
    }"
  >
    <div
      v-for="(cell, index) in props.board"
      :key="index"
      class="relative bg-orange-200 border border-orange-500 aspect-square"
      :style="{
        ...(index % props.boardSize === props.boardSize - 1 && {
          border: '0',
          background: 'transparent',
        }),
        ...(index >= props.board.length - props.boardSize && {
          border: '0',
          background: 'transparent',
        }),
      }"
    >
      <GoStone
        :playerId="cell"
        @click="
          () => {
            if (cell < 0)
              emit(
                'move',
                (props.boardSize - 1 - Math.floor(index / props.boardSize)) * props.boardSize +
                  (index % props.boardSize)
              )
          }
        "
      />
    </div>
  </div>
</template>
