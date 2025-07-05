/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  GoogleGenAI,
  Content,
  Part,
} from '@google/genai';
import OpenAI from 'openai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { getEffectiveModel } from './modelCheck.js';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  SELF_HOSTED_OPENAI = 'self-hosted-openai',
}

export type ContentGeneratorConfig = {
  model: string;
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType | undefined;
  selfHostedEndpoint?: string;
  selfHostedApiKey?: string;
};

export async function createContentGeneratorConfig(
  model: string | undefined,
  authType: AuthType | undefined,
  config?: { getModel?: () => string },
): Promise<ContentGeneratorConfig> {
  const geminiApiKey = process.env.GEMINI_API_KEY;
  const googleApiKey = process.env.GOOGLE_API_KEY;
  const googleCloudProject = process.env.GOOGLE_CLOUD_PROJECT;
  const googleCloudLocation = process.env.GOOGLE_CLOUD_LOCATION;

  // Use runtime model from config if available, otherwise fallback to parameter or default
  const effectiveModel = config?.getModel?.() || model || DEFAULT_GEMINI_MODEL;

  const contentGeneratorConfig: ContentGeneratorConfig = {
    model: effectiveModel,
    authType,
  };

  // if we are using google auth nothing else to validate for now
  if (authType === AuthType.LOGIN_WITH_GOOGLE) {
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.model = await getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
    );

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    !!googleApiKey &&
    googleCloudProject &&
    googleCloudLocation
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;
    contentGeneratorConfig.model = await getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
    );

    return contentGeneratorConfig;
  }

  if (authType === AuthType.SELF_HOSTED_OPENAI) {
    contentGeneratorConfig.selfHostedEndpoint =
      process.env.SELF_HOSTED_OPENAI_ENDPOINT;
    contentGeneratorConfig.selfHostedApiKey =
      process.env.SELF_HOSTED_OPENAI_API_KEY;
    // Note: We might need to add model validation or selection logic here
    // similar to getEffectiveModel if the self-hosted endpoint supports multiple models.
    // For now, we'll assume the provided model in the config is used directly.
    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  sessionId?: string,
): Promise<ContentGenerator> {
  const version = process.env.CLI_VERSION || process.version;
  const httpOptions = {
    headers: {
      'User-Agent': `GeminiCLI/${version} (${process.platform}; ${process.arch})`,
    },
  };
  if (config.authType === AuthType.LOGIN_WITH_GOOGLE) {
    return createCodeAssistContentGenerator(
      httpOptions,
      config.authType,
      sessionId,
    );
  }

  if (
    config.authType === AuthType.USE_GEMINI ||
    config.authType === AuthType.USE_VERTEX_AI
  ) {
    const googleGenAI = new GoogleGenAI({
      apiKey: config.apiKey === '' ? undefined : config.apiKey,
      vertexai: config.vertexai,
      httpOptions,
    });

    return googleGenAI.models;
  }

  // Fallback or error for unhandled auth types
  if (config.authType === AuthType.SELF_HOSTED_OPENAI) {
    if (!config.selfHostedEndpoint) {
      throw new Error(
        'SELF_HOSTED_OPENAI auth type requires SELF_HOSTED_OPENAI_ENDPOINT to be set.',
      );
    }

    try {
      new URL(config.selfHostedEndpoint);
    } catch {
      throw new Error('Invalid self-hosted endpoint URL format');
    }
    
    return new SelfHostedOpenAIContentGenerator(
      config.selfHostedApiKey,
      config.selfHostedEndpoint,
      config.model, // Use the model specified in the config
    );
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported or unconfigured authType: ${config.authType}`,
  );
}

// --- Self-Hosted OpenAI Content Generator ---

// Helper to convert Gemini content to OpenAI messages
function geminiContentToOpenAIMessages(
  geminiContents: GenerateContentParameters['contents'],
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];
  
  // Handle both array and single content
  const contentsArray = Array.isArray(geminiContents) ? geminiContents : [geminiContents];
  
  for (const content of contentsArray) {
    // Cast to Content type to access role and parts
    const contentObj = content as Content;
    
    const role =
      contentObj.role === 'model'
        ? 'assistant'
        : (contentObj.role as OpenAI.Chat.Completions.ChatCompletionRole);
    // Assuming parts is an array of Part objects and we concatenate their text representation.
    // OpenAI API expects a string or an array of content parts (e.g. for images).
    // For simplicity, we'll join text parts here.
    const textContent = (contentObj.parts as Part[])
      .map((part) => ('text' in part ? part.text : ''))
      .join('');
    messages.push({ role, content: textContent } as OpenAI.Chat.Completions.ChatCompletionMessageParam);
  }
  return messages;
}

class SelfHostedOpenAIContentGenerator implements ContentGenerator {
  private openai: OpenAI;
  private model: string;

  constructor(apiKey: string | undefined, baseUrl: string, model: string) {
    this.openai = new OpenAI({
      apiKey: apiKey || undefined, // API key is optional for some self-hosted setups
      baseURL: baseUrl,
    });
    this.model = model;
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = geminiContentToOpenAIMessages(request.contents);
    const completion = await this.openai.chat.completions.create({
      model: this.model,
      messages: messages,
      temperature: request.config?.temperature,
      top_p: request.config?.topP,
      max_tokens: request.config?.maxOutputTokens,
      // TODO: Add support for tools if needed by mapping Gemini tools to OpenAI tools
    });

    // Convert OpenAI response back to Gemini format
    const choices = completion.choices.map((choice) => {
      return {
        message: {
          role: 'model', // Assuming OpenAI assistant role maps to Gemini model role
          parts: [{ text: choice.message?.content || '' }],
        },
        finishReason: choice.finish_reason,
        index: choice.index,
        // TODO: Map other fields like safetyRatings if applicable
      };
    });

    return {
      candidates: choices,
      // TODO: Populate promptFeedback if possible
    } as unknown as GenerateContentResponse; // Type assertion might be needed depending on exact mapping
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = geminiContentToOpenAIMessages(request.contents);
    const stream = await this.openai.chat.completions.create({
      model: this.model,
      messages: messages,
      temperature: request.config?.temperature,
      top_p: request.config?.topP,
      max_tokens: request.config?.maxOutputTokens,
      stream: true,
    });

    async function* generator(): AsyncGenerator<GenerateContentResponse> {
      for await (const part of stream) {
        const choices = part.choices.map((choice) => {
          return {
            message: {
              role: 'model',
              parts: [{ text: choice.delta?.content || '' }],
            },
            finishReason: choice.finish_reason,
            index: choice.index,
          };
        });
        yield {
          candidates: choices,
        } as unknown as GenerateContentResponse;
      }
    }
    return generator();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // OpenAI doesn't have a direct countTokens equivalent for chat models in the same way.
    // This often requires using a tokenizer library like tiktoken.
    // For simplicity, we'll return a placeholder or throw an error.
    // In a real implementation, integrate tiktoken or a similar library.
    console.warn(
      'countTokens is not fully implemented for SelfHostedOpenAIContentGenerator and will return a rough estimate.',
    );
    const textContent = (request.contents as Content[])
      .flatMap((c) => c.parts)
      .filter((p): p is Part => p !== undefined)
      .map((p: Part) => ('text' in p ? p.text : ''))
      .join(' ');
    // Rough estimate: 1 token ~ 4 chars in English
    return { totalTokens: Math.ceil(textContent.length / 4) };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Assuming the self-hosted OpenAI setup has an embedding model endpoint
    // that's compatible or a specific model name for embeddings.
    // This might need adjustment based on the actual vLLM setup.
    if (!this.openai.embeddings) {
        throw new Error("Embeddings are not configured for this OpenAI client.");
    }
    const texts = request.contents as string[]; // Assuming contents are strings for embedding
    const embeddingResponse = await this.openai.embeddings.create({
      model: request.model || 'text-embedding-ada-002', // Or a model configured for vLLM
      input: texts,
    });

    const embeddings = embeddingResponse.data.map((emb) => ({
      values: emb.embedding,
    }));

    return { embeddings };
  }
}
