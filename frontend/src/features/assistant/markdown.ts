import { marked } from 'marked';
import DOMPurify from 'dompurify';

/**
 * The assistant's answers as safe HTML (markdown: lists, tables, code). One renderer for every place an answer is
 * shown: the Assistant tab, the floating chat and the reply box under the 3D view.
 */
export const renderMarkdown = (text: string): string => DOMPurify.sanitize(marked.parse(text, { async: false, breaks: true }) as string);
