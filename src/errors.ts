/**
 * Custom error types for Memorandum MCP server.
 */

/** Base error class for all Memorandum domain errors. */
export class MemorandumError extends Error {
  readonly code: string;
  readonly details?: Record<string, unknown>;

  /**
   * @param code - Machine-readable error code (e.g. "not_found").
   * @param message - Human-readable error description.
   * @param details - Optional structured metadata about the error.
   */
  constructor(code: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'MemorandumError';
    this.code = code;
    this.details = details;
  }

  /** Serialize this error into an MCP-compatible plain object. */
  toMcpError(): Record<string, unknown> {
    return {
      error: this.code,
      message: this.message,
      ...(this.details ? { details: this.details } : {}),
    };
  }
}

/**
 * Convert an error to an MCP-compatible error response.
 */
export function toMcpErrorResponse(error: unknown): {
  content: Array<{ type: 'text'; text: string }>;
  isError: true;
} {
  if (error instanceof MemorandumError) {
    return {
      content: [{ type: 'text' as const, text: JSON.stringify(error.toMcpError()) }],
      isError: true,
    };
  }
  if (error instanceof Error) {
    return {
      content: [{ type: 'text' as const, text: JSON.stringify({ error: 'internal_error', message: error.message }) }],
      isError: true,
    };
  }
  throw error;
}
