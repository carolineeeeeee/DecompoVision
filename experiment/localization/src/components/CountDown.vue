<template>
  <el-progress v-if="props.isProgressBar" :percentage="Math.round(progress * 100)" />
  <h3 v-else>{{ countdown }}</h3>
</template>
<script setup lang="ts">
import { ref, watch, computed, onMounted } from 'vue';
const emit = defineEmits(['timeup']);
const props = defineProps<{ expTime: number; isProgressBar?: boolean }>();
const countdown = ref<string>('0 min, 0 sec, 0 ms');
const timeupEmitted = ref(false); // indicates whether timeup event has been emited, only emits once

// const timeLeft = computed(() => props.expTime - Date.now());
const timeLeft = ref(0);
const totalTime = ref(0);
const progress = computed(() => Math.min(((totalTime.value - timeLeft.value) / totalTime.value), 1));

type Diff = {
  diffDays: number;
  diffHrs: number;
  diffMins: number;
  diffSecs: number;
  diffMs: number;
};

const getDiffDate = (diffMs: number): Diff => {
  let diffSecs = Math.floor(diffMs / 1000);
  const numSecADay = 24 * 60 * 60;
  const diffDays = Math.floor(diffSecs / numSecADay);
  const diffHrs = Math.floor((diffSecs % numSecADay) / 3600);
  const diffMins = Math.floor(((diffSecs % numSecADay) % 3600) / 60);
  diffSecs = ((diffSecs % numSecADay) % 3600) % 60;
  return {
    diffDays,
    diffHrs,
    diffMins,
    diffSecs,
    diffMs: diffMs % 1000,
  };
};

const getDiffMsg = (diff: Diff) => {
  let msg = '';
  if (diff.diffDays) msg += `${diff.diffDays} days, `;
  if (diff.diffHrs) msg += `${diff.diffHrs} hours, `;
  if (diff.diffMins) msg += `${diff.diffMins} minutes, `;
  if (diff.diffSecs) msg += `${diff.diffSecs} seconds, `;
  if (diff.diffMs) msg += `${diff.diffMs} ms`;
  return msg;
};

const updateCountdown = () => {
  const now = Date.now();
  timeLeft.value = props.expTime - now;
  const diff = getDiffDate(timeLeft.value);
  const isExpired = props.expTime < now;
  if (isExpired) {
    countdown.value = '0 ms';
    // emit a timeup event when time is up, emit only once to avoid confusion
    if (!timeupEmitted.value) {
      emit('timeup');
      timeupEmitted.value = true;
    }
  } else {
    countdown.value = getDiffMsg(diff);
    timeLeft.value = props.expTime - now;
  }
};

// When expire time is updated, reset timeupEmitted
watch(
  () => props.expTime,
  (val: number) => {
    timeupEmitted.value = false;
    totalTime.value = val - Date.now();
  }
);

onMounted(() => {
  totalTime.value = props.expTime - Date.now();
});

updateCountdown();
setInterval(() => {
  updateCountdown();
}, 200);
</script>
