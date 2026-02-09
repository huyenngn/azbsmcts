import { PlayerColor, type PreviousMoveInfo } from '@/lib/types'

export function parseBoard(observation: string): number[] {
  const rows = observation.matchAll(/\d\s([+OX]+)/g)
  const newBoard: number[] = []
  for (const row of rows) {
    newBoard.push(...Array.from(row[1]).map((cell) => (cell === '+' ? -1 : cell === 'O' ? 1 : 0)))
  }
  return newBoard
}

export function formatMoveInfo(info: PreviousMoveInfo): string {
  const player = info.player === PlayerColor.Black ? 'Black' : 'White'
  if (info.was_pass) {
    return `${player} passed.`
  } else if (info.captured_stones > 0) {
    return `${player} captured ${info.captured_stones} ${info.captured_stones > 1 ? 'stones' : 'stone'}.`
  }
  return `${player}'s move was ${info.was_observational ? 'observational' : 'valid'}.`
}
