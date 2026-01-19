<script setup lang="ts">
import { PlayerColor } from '@/lib/types'

const props = defineProps<{
  playerId: number
}>()
</script>

<template>
  <div
    class="absolute rounded-full aspect-square w-8 -top-4 -left-4"
    :class="{
      'cursor-pointer': props.playerId === -1,
      'bg-black shadow-md ease-in-out duration-300': props.playerId === PlayerColor.Black,
      'bg-white shadow-md ease-in-out duration-300': props.playerId === PlayerColor.White,
    }"
  ></div>
</template>
