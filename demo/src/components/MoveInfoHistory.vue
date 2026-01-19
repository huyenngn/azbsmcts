<script setup lang="ts">
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'

const props = defineProps<{
  previousMoveInfos: string[]
}>()
</script>

<template>
  <ScrollArea class="h-65 w-50 rounded-md border">
    <div class="p-4">
      <h6 class="mb-2">Previous Moves</h6>
      <TransitionGroup name="move-info" tag="div" class="flex flex-col-reverse">
        <div v-for="(info, index) in props.previousMoveInfos" :key="`${index}`" class="move-item">
          <Separator />
          <div class="py-2">
            {{ info }}
          </div>
        </div>
      </TransitionGroup>
    </div>
  </ScrollArea>
</template>

<style scoped>
.move-info-enter-active {
  transition: all 0.5s ease-out;
}

.move-info-enter-from {
  opacity: 0;
  transform: translateY(20px) scale(0.95);
}

.move-info-enter-to {
  opacity: 1;
  transform: translateY(0) scale(1);
}

.move-info-leave-active {
  transition: all 0.5s ease-in;
}

.move-info-leave-from {
  opacity: 1;
  transform: translateY(0) scale(1);
}

.move-info-leave-to {
  opacity: 0;
  transform: translateY(20px) scale(0.95);
}

.move-info-move {
  transition: transform 0.3s ease;
}
</style>
