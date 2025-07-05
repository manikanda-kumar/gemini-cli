/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createContentGenerator, AuthType } from './contentGenerator.js';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { GoogleGenAI } from '@google/genai';

vi.mock('../code_assist/codeAssist.js');
vi.mock('@google/genai');

// Mock the OpenAI constructor and its methods
vi.mock('openai', () => {
  const MockOpenAI = vi.fn();
  MockOpenAI.prototype.chat = {
    completions: {
      create: vi.fn(),
    },
  };
  MockOpenAI.prototype.embeddings = {
    create: vi.fn(),
  };
  return { default: MockOpenAI };
});

describe('contentGenerator', () => {
  const OLD_ENV = process.env;

  beforeEach(() => {
    vi.resetModules(); // Most important - it clears the cache
    process.env = { ...OLD_ENV }; // Make a copy
  });

  afterEach(() => {
    process.env = OLD_ENV; // Restore old environment
    vi.clearAllMocks();
  });

  it('should create a CodeAssistContentGenerator', async () => {
    const mockGenerator = {} as unknown;
    vi.mocked(createCodeAssistContentGenerator).mockResolvedValue(
      mockGenerator as never,
    );
    const generator = await createContentGenerator({
      model: 'test-model',
      authType: AuthType.LOGIN_WITH_GOOGLE,
    });
    expect(createCodeAssistContentGenerator).toHaveBeenCalled();
    expect(generator).toBe(mockGenerator);
  });

  it('should create a GoogleGenAI content generator', async () => {
    const mockGenerator = {
      models: {},
    } as unknown;
    vi.mocked(GoogleGenAI).mockImplementation(() => mockGenerator as never);
    const generator = await createContentGenerator({
      model: 'test-model',
      apiKey: 'test-api-key',
      authType: AuthType.USE_GEMINI,
    });
    expect(GoogleGenAI).toHaveBeenCalledWith({
      apiKey: 'test-api-key',
      vertexai: undefined,
      httpOptions: {
        headers: {
          'User-Agent': expect.any(String),
        },
      },
    });
    expect(generator).toBe((mockGenerator as GoogleGenAI).models);
  });

  // --- Tests for SELF_HOSTED_OPENAI ---
  describe('createContentGeneratorConfig (Self-Hosted)', () => {
    it('should populate selfHostedEndpoint and selfHostedApiKey from env vars for SELF_HOSTED_OPENAI', async () => {
      process.env.SELF_HOSTED_OPENAI_ENDPOINT = 'http://localhost:8080';
      process.env.SELF_HOSTED_OPENAI_API_KEY = 'test-key';
      // Need to import createContentGeneratorConfig here because of vi.resetModules()
      const { createContentGeneratorConfig } = await import(
        './contentGenerator.js'
      );

      const config = await createContentGeneratorConfig(
        'gpt-3.5-turbo',
        AuthType.SELF_HOSTED_OPENAI,
      );

      expect(config.authType).toBe(AuthType.SELF_HOSTED_OPENAI);
      expect(config.selfHostedEndpoint).toBe('http://localhost:8080');
      expect(config.selfHostedApiKey).toBe('test-key');
      expect(config.model).toBe('gpt-3.5-turbo');
    });

    it('should return undefined for selfHostedApiKey if env var is not set', async () => {
      process.env.SELF_HOSTED_OPENAI_ENDPOINT = 'http://localhost:8080';
      // SELF_HOSTED_OPENAI_API_KEY is not set
      const { createContentGeneratorConfig } = await import(
        './contentGenerator.js'
      );

      const config = await createContentGeneratorConfig(
        'gpt-3.5-turbo',
        AuthType.SELF_HOSTED_OPENAI,
      );

      expect(config.selfHostedEndpoint).toBe('http://localhost:8080');
      expect(config.selfHostedApiKey).toBeUndefined();
    });

    it('should not populate self-hosted fields for USE_GEMINI auth type', async () => {
      process.env.GEMINI_API_KEY = 'gemini-api-key';
      const { createContentGeneratorConfig } = await import(
        './contentGenerator.js'
      );
      const config = await createContentGeneratorConfig(
        'gemini-pro',
        AuthType.USE_GEMINI,
      );
      expect(config.authType).toBe(AuthType.USE_GEMINI);
      expect(config.selfHostedEndpoint).toBeUndefined();
      expect(config.selfHostedApiKey).toBeUndefined();
    });
  });

  describe('createContentGenerator (Self-Hosted)', () => {
    it('should return SelfHostedOpenAIContentGenerator for SELF_HOSTED_OPENAI auth type with endpoint', async () => {
      const { createContentGenerator } = await import('./contentGenerator.js');
      const { default: OpenAIConstructor } = await import('openai');


      const config = {
        model: 'gpt-3.5-turbo',
        authType: AuthType.SELF_HOSTED_OPENAI,
        selfHostedEndpoint: 'http://localhost:8080/v1',
        selfHostedApiKey: 'test-api-key',
      };

      const generator = await createContentGenerator(
        config as import('./contentGenerator.js').ContentGeneratorConfig,
      );
      expect(generator).toBeDefined();
      expect((generator as any).openai).toBeDefined(); // Check for our specific implementation detail
      expect(OpenAIConstructor).toHaveBeenCalledWith({
        apiKey: 'test-api-key',
        baseURL: 'http://localhost:8080/v1',
      });
    });

    it('should throw an error if SELF_HOSTED_OPENAI is used without an endpoint', async () => {
      const { createContentGenerator } = await import('./contentGenerator.js');
      const config = {
        model: 'gpt-3.5-turbo',
        authType: AuthType.SELF_HOSTED_OPENAI,
        // selfHostedEndpoint is missing
      };

      await expect(
        createContentGenerator(
          config as import('./contentGenerator.js').ContentGeneratorConfig,
        ),
      ).rejects.toThrow(
        'SELF_HOSTED_OPENAI auth type requires SELF_HOSTED_OPENAI_ENDPOINT to be set.',
      );
    });

    it('should use undefined API key for SelfHostedOpenAIContentGenerator if not provided', async () => {
      const { createContentGenerator } = await import('./contentGenerator.js');
      const { default: OpenAIConstructor } = await import('openai');

      const config = {
        model: 'gpt-3.5-turbo',
        authType: AuthType.SELF_HOSTED_OPENAI,
        selfHostedEndpoint: 'http://localhost:8080/v1',
        // selfHostedApiKey is undefined
      };
      await createContentGenerator(
        config as import('./contentGenerator.js').ContentGeneratorConfig,
      );
      expect(OpenAIConstructor).toHaveBeenCalledWith({
        apiKey: undefined,
        baseURL: 'http://localhost:8080/v1',
      });
    });
  });
});
