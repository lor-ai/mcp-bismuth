import { TitanMemoryServer } from '../index.js';

describe('TitanMemoryServer', () => {
  let server: TitanMemoryServer;

  beforeEach(async () => {
    server = new TitanMemoryServer();
  });

  afterEach(async () => {
    if (server) {
      // Clean shutdown - the server doesn't have a dispose method in the interface
    }
  });

  test('initializes correctly', () => {
    expect(server).toBeDefined();
  });

  test('handles basic operations', async () => {
    // Basic test that doesn't require complex setup
    expect(server).toBeInstanceOf(TitanMemoryServer);
  });
});
