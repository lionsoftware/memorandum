/**
 * Logger setup for Memorandum MCP server.
 *
 * Uses pino with stderr transport (stdout is reserved for MCP protocol).
 */

import pino from 'pino';

/**
 * Create a pino logger that writes to stderr.
 * @param level - Log level (default: "info").
 * @returns Configured pino logger instance.
 */
export function createLogger(level: string = 'info'): pino.Logger {
  return pino({
    name: 'memorandum',
    level,
    transport: {
      target: 'pino/file',
      options: { destination: 2 }, // stderr
    },
  });
}
