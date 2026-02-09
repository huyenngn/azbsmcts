import { type GameStateResponse } from '@/lib/types'

type StreamHandlers = {
  onUpdate: (data: GameStateResponse) => void
  onError?: (data: GameStateResponse) => void
}

export async function streamGameUpdates(
  url: string,
  init: RequestInit,
  handlers: StreamHandlers,
): Promise<void> {
  const response = await fetch(url, init)

  if (!response.body) {
    throw new Error('Streaming response not supported')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''

    for (const part of parts) {
      const lines = part.split('\n')
      let eventType = 'message'
      let dataStr = ''

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventType = line.replace('event:', '').trim()
        } else if (line.startsWith('data:')) {
          dataStr += line.replace('data:', '').trim()
        }
      }

      if (!dataStr) continue
      const data = JSON.parse(dataStr) as GameStateResponse

      if (eventType === 'error') {
        handlers.onError?.(data)
        continue
      }

      handlers.onUpdate(data)
    }
  }
}

export async function getBackend<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(response.statusText)
  }

  return response.json() as Promise<T>
}
